import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import Transformer
from loguru import logger as loguru_logger


class MISSRec(Transformer):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config["train_stage"]
        self.temperature = config["temperature"]
        self.lam = config["lambda"]
        self.gamma = config["gamma"]
        self.modal_type = config["modal_type"]
        self.id_type = config["id_type"]
        self.seq_mm_fusion = config["seq_mm_fusion"]  # 'add' | 'contextual'
        assert self.seq_mm_fusion in ["add", "contextual"]
        self.item_mm_fusion = config[
            "item_mm_fusion"
        ]  # 'static' | 'dynamic_shared' | 'dynamic_instance'
        # NOTE: `plm_embedding` in pre-train stage will be carried via dataloader
        assert self.item_mm_fusion in ["static", "dynamic_shared", "dynamic_instance"]
        self.USER_PROFILE = config["LABEL_FIELD"]

        assert self.train_stage in [
            "pretrain",
            "inductive_ft",
            "transductive_ft",
            "userprofile_ft",
        ], f"Unknown train stage: [{self.train_stage}]"
        # loguru_logger.debug(f"train_stage in MISSRec init: {self.train_stage}")
        if self.train_stage in ["pretrain", "inductive_ft", "userprofile_ft"]:
            self.item_embedding = None  # with id
        # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ["inductive_ft", "transductive_ft", "userprofile_ft"]:
            # NOTE: `plm_embedding` in pre-train stage will be carried via dataloader
            all_num_embeddings = 0
            if "text" in self.modal_type:
                self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
                self.register_buffer(
                    "plm_embedding_empty_mask",
                    (~self.plm_embedding.weight.data.sum(-1).bool()),
                )
                all_num_embeddings += self.plm_embedding.num_embeddings - 1
                self.register_buffer(
                    "plm_interest_lookup_table",
                    torch.zeros(self.plm_embedding.num_embeddings, dtype=torch.long),
                )
            if "img" in self.modal_type:
                self.img_embedding = copy.deepcopy(dataset.img_embedding)
                self.register_buffer(
                    "img_embedding_empty_mask",
                    (~self.img_embedding.weight.data.sum(-1).bool()),
                )
                all_num_embeddings += self.img_embedding.num_embeddings - 1
                self.register_buffer(
                    "img_interest_lookup_table",
                    torch.zeros(self.img_embedding.num_embeddings, dtype=torch.long),
                )

            # NOTE: 只在下游微调时起效
            self.num_interest = max(
                math.ceil(all_num_embeddings * config["interest_ratio"]), 1
            )
            self.knn_local_size = max(
                math.ceil(all_num_embeddings * config["knn_local_ratio"]), 1
            )
            self.register_buffer(
                "interest_embeddings",
                torch.zeros(
                    self.num_interest + 1, config["hidden_size"], dtype=torch.float
                ),
            )

        if "text" in self.modal_type and "img" in self.modal_type:
            if self.item_mm_fusion == "dynamic_shared":
                self.fusion_factor = nn.Parameter(
                    data=torch.tensor(0, dtype=torch.float)
                )
            elif self.item_mm_fusion == "dynamic_instance":
                self.fusion_factor = nn.Parameter(
                    data=torch.zeros(self.n_items, dtype=torch.float)
                )

        if "text" in self.modal_type:
            self.text_adaptor = nn.Linear(config["plm_size"], config["hidden_size"])

        if "img" in self.modal_type:
            self.img_adaptor = nn.Linear(config["img_size"], config["hidden_size"])

        if self.train_stage in ["userprofile_ft"]:
            self.mlp1 = nn.Linear(
                self.plm_embedding.num_embeddings, config["hidden_size"]
            )
            # Activation function
            self.activation1 = nn.ReLU()
            self.mlp2 = nn.Linear(config["hidden_size"], config["num_classes"])
            self.activation2 = nn.Softmax(dim=-1)

    def get_encoder_attention_mask(self, dec_input_seq=None, is_casual=True):
        """memory_mask: [BxL], dec_input_seq: [BxNq]"""
        key_padding_mask = dec_input_seq == 0  # binary, [BxNq], Nq=L
        dec_seq_len = dec_input_seq.size(-1)
        attn_mask = (
            torch.triu(
                torch.full(
                    (dec_seq_len, dec_seq_len),
                    float("-inf"),
                    device=dec_input_seq.device,
                ),
                diagonal=1,
            )
            if is_casual
            else None
        )
        return attn_mask, key_padding_mask

    def get_decoder_attention_mask(
        self, enc_input_seq, item_modal_empty_mask, is_casual=True
    ):
        # enc_input_seq: [BxL]
        # item_modal_empty_mask: [BxMxL]
        assert enc_input_seq.size(0) == item_modal_empty_mask.size(0)
        assert enc_input_seq.size(-1) == item_modal_empty_mask.size(-1)
        batch_size, num_modality, seq_len = item_modal_empty_mask.shape  # M
        if self.seq_mm_fusion == "add":
            key_padding_mask = enc_input_seq == 0  # binary, [BxL]
        else:
            # binary, [Bx1xL] | [BxMxL] => [BxMxL]
            key_padding_mask = torch.logical_or(
                (enc_input_seq == 0).unsqueeze(1), item_modal_empty_mask
            )
            key_padding_mask = key_padding_mask.flatten(1)  # [BxMxL] => [Bx(M*L)]
        if is_casual:
            attn_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), float("-inf"), device=enc_input_seq.device
                ),
                diagonal=1,
            )  # [LxL]
            if self.seq_mm_fusion != "add":
                attn_mask = torch.tile(
                    attn_mask, (num_modality, num_modality)
                )  # [(M*L)x(M*L)]
        else:
            attn_mask = None
        cross_attn_mask = None  # Full mask
        return attn_mask, cross_attn_mask, key_padding_mask

    # def forward(self, enc_item_seq, item_emb, item_modal_empty_mask, item_seq_len, dec_input_seq=None, dec_input_emb=None, dec_inp_seq_len=None):
    def forward(
        self,
        item_seq,
        item_emb,
        item_modal_empty_mask,
        item_seq_len,
        interest_seq=None,
        interest_emb=None,
        interest_seq_len=None,
    ):
        # encoder input
        enc_input_emb = interest_emb
        src_attn_mask, src_key_padding_mask = self.get_encoder_attention_mask(
            interest_seq, is_casual=False
        )

        # decoder input
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        # loguru_logger.debug(f"position_ids.shape: {position_ids.shape}")  # [L]
        position_embedding = self.position_embedding(position_ids)  # [LxD]

        # loguru_logger.debug(
        # f"position_embedding.shape: {position_embedding.shape}"
        # )  # [LxD]
        # loguru_logger.debug(f"item_emb.shape: {item_emb.shape}")  # [BxLxD]

        # repeat position_embedding to match the batch size
        position_embedding = position_embedding.unsqueeze(0).expand(
            item_seq.size(0), -1, -1
        )  # [BxLxD]
        # loguru_logger.debug(
        # f"position_embedding.shape after expand: {position_embedding.shape}"
        # )  # [BxLxD]
        dec_input_emb = item_emb + position_embedding  # [BxMxLxD] or [BxLxD]
        # loguru_logger.debug(f"dec_input_emb.shape: {dec_input_emb.shape}")  # [BxLxD]

        if self.train_stage == "transductive_ft":
            if self.id_type != "none":
                item_id_embeddings = self.item_embedding(item_seq)
                if self.seq_mm_fusion != "add":
                    item_id_embeddings = item_id_embeddings.unsqueeze(1)  # [Bx1xLxD]
                dec_input_emb = dec_input_emb + item_id_embeddings
        if self.seq_mm_fusion != "add":
            dec_input_emb = dec_input_emb.view(
                dec_input_emb.size(0), -1, dec_input_emb.size(-1)
            )  # [BxMxLxD] => [Bx(M*L)xD]
        dec_input_emb = self.LayerNorm(dec_input_emb)
        # dec_input_emb: [BxMxLxD] or [BxLxD]
        dec_input_emb = self.dropout(dec_input_emb)
        # dec_input_emb: [BxMxLxD] or [BxLxD]
        (
            tgt_attn_mask,
            tgt_cross_attn_mask,
            tgt_key_padding_mask,
        ) = self.get_decoder_attention_mask(
            item_seq, item_modal_empty_mask, is_casual=False
        )
        memory_key_padding_mask = src_key_padding_mask

        # contextual encoder
        memory = self.trm_model.encoder(
            src=enc_input_emb,
            mask=src_attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # diversity regularization for interest tokens
        src_key_token_weight = (
            (~src_key_padding_mask).unsqueeze(-1).float().mean(1, keepdim=True)
        )  # [BxL] => [BxLx1] => [Bx1x1]
        pooled_memory = (memory * src_key_token_weight).sum(
            1
        )  # ([BxLxD] * [Bx1x1]) => [BxD]
        interest_orthogonal_regularization = (
            pooled_memory * pooled_memory
        ).sum() / pooled_memory.shape[
            1
        ]  # [BxD] x [BxD] => [B]

        # interest-aware decoder
        trm_output = self.trm_model.decoder(
            dec_input_emb,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=tgt_cross_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        output = self.gather_indexes(trm_output, item_seq_len - 1)
        # output: [BxD]
        # if self.train_stage in ["userprofile_ft"]:
        #     output = self.mlp(output)
        # output: [BxN](目标用户画像的类别数)
        return output, interest_orthogonal_regularization.mean()  # [BxD], []

    def seq_item_contrastive_task(self, seq_output, interaction, batch_labels):
        if "text" in self.modal_type:
            pos_text_emb = self.text_adaptor(interaction["pos_text_emb"])
        if "img" in self.modal_type:
            pos_img_emb = self.img_adaptor(interaction["pos_img_emb"])
        if "text" in self.modal_type and "img" in self.modal_type:  # weighted fusion
            logits = (
                self._compute_dynamic_fused_logits(
                    seq_output, pos_text_emb, pos_img_emb
                )
                / self.temperature
            )
        else:  # single modality or no modality
            if "text" in self.modal_type:
                pos_item_emb = pos_text_emb
            if "img" in self.modal_type:
                pos_item_emb = pos_img_emb
            pos_items_emb = F.normalize(pos_items_emb, dim=1)
            logits = (
                torch.matmul(seq_output, pos_item_emb.transpose(0, 1))
                / self.temperature
            )
        loss = F.cross_entropy(logits, batch_labels)
        return loss

    def seq_seq_contrastive_task(self, seq_output, interaction, img_emb, batch_labels):
        (
            seq_output_aug,
            interest_orthogonal_regularization_aug,
        ) = self._compute_seq_embeddings_pretrain(
            item_seq=interaction[self.ITEM_SEQ + "_aug"],
            item_seq_len=interaction[self.ITEM_SEQ_LEN + "_aug"],
            text_emb=self.text_adaptor(interaction["text_emb_list_aug"]),
            img_emb=img_emb,
            text_emb_empty_mask=interaction["text_emb_empty_mask_list_aug"],
            img_emb_empty_mask=interaction["img_emb_empty_mask_list"],
            # text_interest_seq=interaction['text_interest_list_aug'],
            # img_interest_seq=interaction['img_interest_list'],
            unique_interest_seq=interaction["unique_interest_list_aug"],
            unique_interest_emb_list=interaction["unique_interest_emb_list_aug"],
            unique_interest_len=interaction["unique_interest_len_aug"],
        )
        logits = (
            torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        )
        loss = F.cross_entropy(logits, batch_labels)
        return loss, interest_orthogonal_regularization_aug

    # RecBole给出的Custom Model中只有一个predict函数, 但是MISSRec使用了pretrain和finetune.
    # 是怎么做到把predict换成了pretrain和finetune的呢?
    # 嗷! 并没有换, 而是在calculate_loss中调用了pretrain和finetune, 而在predict则是使用的速度更快的full_sort_predict.
    def pretrain(self, interaction):
        img_emb = self.img_adaptor(interaction["img_emb_list"])
        # Input: []
        # Output: []
        (
            seq_output,
            interest_orthogonal_regularization,
        ) = self._compute_seq_embeddings_pretrain(
            item_seq=interaction[self.ITEM_SEQ],
            item_seq_len=interaction[self.ITEM_SEQ_LEN],
            text_emb=self.text_adaptor(interaction["text_emb_list"]),
            img_emb=img_emb,
            text_emb_empty_mask=interaction["text_emb_empty_mask_list"],
            img_emb_empty_mask=interaction["img_emb_empty_mask_list"],
            # text_interest_seq=interaction['text_interest_list'],
            # img_interest_seq=interaction['img_interest_list'],
            unique_interest_seq=interaction["unique_interest_list"],
            unique_interest_emb_list=interaction["unique_interest_emb_list"],
            unique_interest_len=interaction["unique_interest_len"],
        )
        batch_size = seq_output.shape[0]
        device = seq_output.device
        batch_labels = torch.arange(batch_size, device=device, dtype=torch.long)

        loss_seq_item = self.seq_item_contrastive_task(
            seq_output, interaction, batch_labels
        )
        (
            loss_seq_seq,
            interest_orthogonal_regularization_aug,
        ) = self.seq_seq_contrastive_task(
            seq_output, interaction, img_emb, batch_labels
        )
        loss = (
            loss_seq_item
            + self.lam * loss_seq_seq
            + self.gamma
            * (
                interest_orthogonal_regularization
                + interest_orthogonal_regularization_aug
            )
        )
        return loss

    def userprofile(self, interaction):
        # loguru_logger.debug(f"Enter userprofile")
        # Loss for fine-tuning
        # loguru_logger.debug(f"interaction: {interaction}")
        # loguru_logger.debug(f"self.USER_PROFILE: {self.USER_PROFILE}")
        # loguru_logger.debug(
        # f"shape of interaction[self.USER_PROFILE]: {interaction[self.USER_PROFILE].shape}"
        # )  # [B]
        # loguru_logger.debug(f"self.ITEM_SEQ: {self.ITEM_SEQ}")  # L
        item_seq = interaction[self.ITEM_SEQ]  # [BxL]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]

        # loguru_logger.debug(f"item_seq shape: {item_seq.shape}")
        # loguru_logger.debug(f"item_seq_len shape: {item_seq_len.shape}")

        seq_output, interest_orthogonal_regularization = self._compute_seq_embeddings(
            item_seq, item_seq_len
        )
        # loguru_logger.debug(f"seq_output shape: {seq_output.shape}")
        # loguru_logger.debug(
        # f"interest_orthogonal_regularization shape: {interest_orthogonal_regularization.shape}"
        # )
        if "text" in self.modal_type and "img" in self.modal_type:  # weighted fusion
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            logits = (
                self._compute_dynamic_fused_logits(
                    seq_output, test_text_emb, test_img_emb
                )
                / self.temperature
            )
        else:  # single modality or no modality
            test_item_emb = self._compute_test_item_embeddings()
            logits = (
                torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                / self.temperature
            )
            # loguru_logger.debug(f"logits shape: {logits.shape}")
            logits = self.mlp1(logits)
            logits = self.activation1(logits)
            logits = self.mlp2(logits)
            logits = self.activation2(logits)
            # logits: [BxN]
            # remain the big one in the logits
            # logits = logits.max(dim=-1).values
            # loguru_logger.debug(f"logits shape after mlp: {logits.shape}")
            # loguru_logger.debug(f"logits: {logits}")
        # pos_items should be the user profile
        # pos_items = interaction[self.POS_ITEM_ID]
        pos_items = interaction[self.USER_PROFILE] - 1
        # loguru_logger.debug(f"pos_items shape: {pos_items.shape}")
        # loguru_logger.debug(f"pos_items: {pos_items}")
        # DEBUG: Show the logits shape and the pos_items shape
        # loguru_logger.debug(logits.shape, pos_items.shape)
        loss = (
            self.loss_fct(logits, pos_items)
            + self.gamma * interest_orthogonal_regularization
        )
        return loss

    def _compute_seq_embeddings_pretrain(
        self,
        item_seq,
        item_seq_len,
        text_emb,
        img_emb,
        text_emb_empty_mask=None,
        img_emb_empty_mask=None,
        # text_interest_seq=None,
        # img_interest_seq=None
        unique_interest_seq=None,
        unique_interest_emb_list=None,
        unique_interest_len=None,
    ):
        item_emb_list = 0 if self.seq_mm_fusion == "add" else []
        item_modal_empty_mask_list = []
        # interest_seq_list = []
        if "text" in self.modal_type:
            if self.seq_mm_fusion == "add":
                item_emb_list = item_emb_list + text_emb
            else:
                item_emb_list.append(text_emb)  # append [BxLxD]
            item_modal_empty_mask_list.append(text_emb_empty_mask)  # append [BxL]
            # interest_seq_list.append(text_interest_seq)
        if "img" in self.modal_type:
            if self.seq_mm_fusion == "add":
                item_emb_list = item_emb_list + img_emb  # [BxLxD]
            else:
                item_emb_list.append(img_emb)  # append [BxLxD]
            item_modal_empty_mask_list.append(img_emb_empty_mask)  # append [BxL]
            # interest_seq_list.append(img_interest_seq)
        if self.seq_mm_fusion != "add":
            item_emb_list = torch.stack(item_emb_list, dim=1)  # [BxMxLxD]
        item_modal_empty_mask = torch.stack(
            item_modal_empty_mask_list, dim=1
        )  # [BxMxL]

        seq_output, interest_orthogonal_regularization = self.forward(
            item_seq=item_seq,
            item_emb=item_emb_list,
            item_modal_empty_mask=item_modal_empty_mask,
            item_seq_len=item_seq_len,
            interest_seq=unique_interest_seq,
            interest_emb=unique_interest_emb_list,
            interest_seq_len=unique_interest_len,
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output, interest_orthogonal_regularization

    def _compute_seq_embeddings(self, item_seq, item_seq_len):
        # loguru_logger.debug(f"shape of item_seq: {item_seq.shape}")  # [BxL]
        # loguru_logger.debug(f"shape of item_seq_len: {item_seq_len.shape}")  # [B]
        if "text" in self.modal_type:
            tmp_plm_embedding = self.plm_embedding(item_seq)
            # loguru_logger.debug(
            # f"shape of tmp_plm_embedding: {tmp_plm_embedding.shape}"
            # )
            text_emb = self.text_adaptor(tmp_plm_embedding)
            # item_seq: [BxL]
            # plm_embedding: [BxLxD]
            # text_emb: [BxLxD]
            # loguru_logger.debug(f"shape of text_emb: {text_emb.shape}")
            text_emb_empty_mask = self.plm_embedding_empty_mask[item_seq]
            # loguru_logger.debug(
            # f"shape of text_emb_empty_mask: {text_emb_empty_mask.shape}"
            # )
            # text_emb_empty_mask: [BxL]
        if "img" in self.modal_type:
            img_emb = self.img_adaptor(self.img_embedding(item_seq))
            img_emb_empty_mask = self.img_embedding_empty_mask[item_seq]

        # 改进4：把加法改成stack
        item_emb_list = 0 if self.seq_mm_fusion == "add" else []
        item_modal_empty_mask_list = []
        interest_seq_list = []
        if "text" in self.modal_type:
            if self.seq_mm_fusion == "add":
                item_emb_list = item_emb_list + text_emb
                # item_emb_list: [BxLxD]
            else:
                item_emb_list.append(text_emb)  # append [BxLxD]
            item_modal_empty_mask_list.append(text_emb_empty_mask)  # append [BxL]
            plm_interest_seq = self.plm_interest_lookup_table[item_seq]  # [BxNq], Nq=L
            interest_seq_list.append(plm_interest_seq)
        if "img" in self.modal_type:
            if self.seq_mm_fusion == "add":
                item_emb_list = item_emb_list + img_emb  # [BxLxD]
            else:
                item_emb_list.append(img_emb)  # append [BxLxD]
            item_modal_empty_mask_list.append(img_emb_empty_mask)  # append [BxL]
            img_interest_seq = self.img_interest_lookup_table[item_seq]  # [BxNq], Nq=L
            interest_seq_list.append(img_interest_seq)
        if self.seq_mm_fusion != "add":
            item_emb_list = torch.stack(item_emb_list, dim=1)  # [BxMxLxD]
        item_modal_empty_mask = torch.stack(
            item_modal_empty_mask_list, dim=1
        )  # [BxMxL]

        # deduplication
        unique_interest_seq = []
        unique_interest_len = []
        all_interest_seq = torch.cat(interest_seq_list, dim=-1)
        for sample in all_interest_seq:
            unique_interests = sample.unique()
            unique_interest_len.append(len(unique_interests))
            unique_interest_seq.append(unique_interests)
        unique_interest_seq = nn.utils.rnn.pad_sequence(
            unique_interest_seq, batch_first=True, padding_value=0
        )
        unique_interest_emb_list = self.interest_embeddings[
            unique_interest_seq
        ]  # [BxNqxD], Nq=L
        unique_interest_len = torch.tensor(
            unique_interest_len, device=unique_interest_seq.device
        )
        del interest_seq_list

        seq_output, interest_orthogonal_regularization = self.forward(
            item_seq=item_seq,
            item_emb=item_emb_list,
            item_modal_empty_mask=item_modal_empty_mask,
            item_seq_len=item_seq_len,
            interest_seq=unique_interest_seq,
            interest_emb=unique_interest_emb_list,
            interest_seq_len=unique_interest_len,
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output, interest_orthogonal_regularization

    def _compute_test_item_embeddings(self):
        test_item_emb = 0
        if "text" in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_item_emb = test_item_emb + test_text_emb
        if "img" in self.modal_type:
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            test_item_emb = test_item_emb + test_img_emb

        if self.train_stage == "transductive_ft":
            if self.id_type != "none":
                test_item_emb = test_item_emb + self.item_embedding.weight

        test_item_emb = F.normalize(test_item_emb, dim=1)
        # loguru_logger.debug(f"test_item_emb shape: {test_item_emb.shape}")
        # loguru_logger.debug(f"test_item_emb: {test_item_emb}")
        return test_item_emb

    def _compute_dynamic_fused_logits(self, seq_output, text_emb, img_emb):
        text_emb = F.normalize(text_emb, dim=1)
        img_emb = F.normalize(img_emb, dim=1)
        text_logits = torch.matmul(seq_output, text_emb.transpose(0, 1))  # [BxB]
        img_logits = torch.matmul(seq_output, img_emb.transpose(0, 1))  # [BxB]
        modality_logits = torch.stack([text_logits, img_logits], dim=-1)  # [BxBx2]
        if self.item_mm_fusion in ["dynamic_shared", "dynamic_instance"]:
            agg_logits = (
                modality_logits
                * F.softmax(modality_logits * self.fusion_factor.unsqueeze(-1), dim=-1)
            ).sum(
                dim=-1
            )  # [BxBx2] => [BxB]
        else:  # 'static'
            agg_logits = modality_logits.mean(dim=-1)  # [BxBx2] => [BxB]
        if self.train_stage == "transductive_ft":
            if self.id_type != "none":
                test_id_emb = F.normalize(self.item_embedding.weight, dim=1)
                id_logits = torch.matmul(seq_output, test_id_emb.transpose(0, 1))
                agg_logits = (id_logits + agg_logits * 2) / 3
        return agg_logits

    def calculate_loss(self, interaction):
        # loguru_logger.debug(f"self.train_stage: {self.train_stage}")
        if self.train_stage == "pretrain":
            return self.pretrain(interaction)
        if self.train_stage == "userprofile_ft":
            return self.userprofile(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, interest_orthogonal_regularization = self._compute_seq_embeddings(
            item_seq, item_seq_len
        )
        if "text" in self.modal_type and "img" in self.modal_type:  # weighted fusion
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            logits = (
                self._compute_dynamic_fused_logits(
                    seq_output, test_text_emb, test_img_emb
                )
                / self.temperature
            )
        else:  # single modality or no modality
            test_item_emb = self._compute_test_item_embeddings()
            logits = (
                torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                / self.temperature
            )

        pos_items = interaction[self.POS_ITEM_ID]
        loss = (
            self.loss_fct(logits, pos_items)
            + self.gamma * interest_orthogonal_regularization
        )
        return loss

    # TODO: Need update for userprofile task
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, _ = self._compute_seq_embeddings(item_seq, item_seq_len)
        if "text" in self.modal_type and "img" in self.modal_type:  # weighted fusion
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            scores = (
                self._compute_dynamic_fused_logits(
                    seq_output, test_text_emb, test_img_emb
                )
                / self.temperature
            )
        else:  # single modality or no modality
            test_item_emb = self._compute_test_item_embeddings()
            scores = (
                torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                / self.temperature
            )
        return scores

    def predict(self, interaction):
        # loguru_logger.debug(f"interaction: {interaction}")
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]

        # loguru_logger.debug(f"item_seq shape: {item_seq.shape}")
        # loguru_logger.debug(f"item_seq_len shape: {item_seq_len.shape}")

        seq_output, interest_orthogonal_regularization = self._compute_seq_embeddings(
            item_seq, item_seq_len
        )
        test_item_emb = self._compute_test_item_embeddings()
        logits = (
            torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        )
        # loguru_logger.debug(f"logits shape: {logits.shape}")
        logits = self.mlp1(logits)
        logits = self.activation1(logits)
        logits = self.mlp2(logits)
        logits = self.activation2(logits)
        # logits: [BxN]
        # remain the big one in the logits
        # logits = logits.max(dim=-1).values
        # loguru_logger.debug(f"logits shape after mlp: {logits.shape}")
        # loguru_logger.debug(f"logits: {logits}")
        # pos_items should be the user profile
        # pos_items = interaction[self.POS_ITEM_ID]
        pos_items = interaction[self.USER_PROFILE] - 1
        # Get the index of the biggest value in the logits
        logits = logits.argmax(dim=-1)
        correct = (logits == pos_items).float()
        return correct
