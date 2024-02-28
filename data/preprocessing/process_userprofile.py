import argparse
import collections
import datetime
import os
import random
from tqdm import tqdm

from process_amazon import (
    filter_inters,
    make_inters_in_order,
    get_user_item_from_ratings,
    generate_item_embedding,
)
from utils import check_path, set_device, load_plm


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, "r") as fp:
        cr = fp.readlines()
        for line in tqdm(cr, desc="Load ratings"):
            try:
                userid, itemid, rating, timestamp = line.strip().split("::")
                users.add(userid)
                items.add(itemid)
                ts = timestamp
                # ts = datetime.datetime.strptime(timestamp).timestamp()
                inters.add((userid, itemid, rating, int(ts)))
            except ValueError:
                print(line)
    return users, items, inters


def preprocess_rating(args):
    print("Process rating data: ")
    print(" Dataset: ", args.dataset)

    # load ratings
    rating_file_path = os.path.join(args.input_path, "ratings.dat")
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print("The number of raw inters: ", len(rating_inters))
    rating_inters = filter_inters(
        rating_inters,
        can_items=rating_items,
        user_k_core_threshold=args.user_k,
        item_k_core_threshold=args.item_k,
    )

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print("\n")

    return rating_inters


def generate_text(args, items):
    item_text_list = []
    meta_file_path = os.path.join(args.input_path, "movies.dat")
    item2text = {}
    with open(meta_file_path, "r", encoding="latin2") as fp:
        cr = fp.readlines()
        for line in tqdm(cr, desc="Load ratings"):
            try:
                (
                    itemid,
                    title,
                    genres,
                ) = line.strip().split("::")
                if itemid not in item2text:
                    item2text[itemid] = title.join(" ".join(genres))
            except ValueError:
                print(line)

    for iid in tqdm(items, desc="Generate text"):
        assert iid in item2text
        text = item2text[iid].strip().lower() + "."
        item_text_list.append([iid, text])
    return item_text_list


def preprocess_text(args, rating_inters):
    print("Process text data: ")
    print(" Dataset: ", args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_text_list = generate_text(args, rating_items)
    print("\n")

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_training_data(args, rating_inters, userprofile):
    print("Split dataset: ")
    print(" Dataset: ", args.dataset)

    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters = dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # Add the user profile
        train_inters[u_index] = []
        for i_index in inters:
            train_inters[u_index].append(str(i_index))
        for up in userprofile[u_index]:
            train_inters[u_index].append(up)
    return train_inters, user2index, item2index


def preprocess_userprofile(args):
    print("Process user profile data: ")
    print(" Dataset: ", args.dataset)
    userprofile = collections.defaultdict(list)
    userprofile_file_path = os.path.join(args.input_path, "users.dat")
    with open(userprofile_file_path, "r") as fp:
        cr = fp.readlines()
        for line in tqdm(cr, desc="Load user profile"):
            try:
                userid, gender, age, occupation, zip = line.strip().split("::")
                age = int(age)
                gender = 0 if gender == "F" else 1
                occupation = int(occupation)
                userprofile[userid].append(age)
                userprofile[userid].append(gender)
                userprofile[userid].append(occupation)
            except ValueError:
                print(line)
    return userprofile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="OR")
    parser.add_argument("--user_k", type=int, default=5, help="user k-core filtering")
    parser.add_argument("--item_k", type=int, default=5, help="item k-core filtering")
    parser.add_argument("--input_path", type=str, default="../raw/")
    parser.add_argument("--output_path", type=str, default="../downstream/")
    parser.add_argument("--gpu_id", type=int, default=2, help="ID of running GPU")
    parser.add_argument("--plm_name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--emb_type",
        type=str,
        default="CLS",
        help="item text emb type, can be CLS or Mean",
    )
    parser.add_argument(
        "--word_drop_ratio",
        type=float,
        default=-1,
        help="word drop ratio, do not drop by default",
    )
    return parser.parse_args()


def convert_to_atomic_files(
    args, train_data, train_radio=0.8, valid_radio=0.1, test_radio=0.1
):
    # train_data: {uid: [iid1, iid2, ..., age, gender, occupation]}
    print("Convert dataset: ")
    print(" Dataset: ", args.dataset)
    uid_list = list(train_data.keys())
    # uid_list.sort(key=lambda t: int(t))
    # shuffle the uid list
    random.shuffle(uid_list)
    inter_list = list()
    # generate the list
    for uid in uid_list:
        train_data[uid] = train_data[uid][:-3]
        userprofile_data = train_data[uid][-3:]
        tmp_list = list()
        tmp_list.append(uid)
        for i in range(len(train_data[uid])):
            for j in range(1, i):
                tmp_list.append(train_data[uid][j])
            # add userprofile data
            tmp_list.extend(userprofile_data)
        inter_list.append(tmp_list)
    train_len = int(len(inter_list) * train_radio)
    valid_len = int(len(inter_list) * valid_radio)
    test_len = len(inter_list) - train_len - valid_len
    print("Train: ", train_len)
    print("Valid: ", valid_len)
    print("Test: ", test_len)

    with open(
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.train.inter"), "w"
    ) as file:
        file.write(
            "user_id:token\titem_id_list:token_seq\tage:token\tgender:token\toccupation:token\n"
        )
        for i in range(train_len):
            file.write(
                f'{inter_list[i][0]}\t{" ".join(inter_list[i][1:-3])}\t{inter_list[i][-3]}\t{inter_list[i][-2]}\t{inter_list[i][-1]}\n'
            )

    with open(
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.valid.inter"), "w"
    ) as file:
        file.write(
            "user_id:token\titem_id_list:token_seq\tage:token\tgender:token\toccupation:token\n"
        )
        for i in range(train_len, train_len + valid_len):
            file.write(
                f'{inter_list[i][0]}\t{" ".join(inter_list[i][1:-3])}\t{inter_list[i][-3]}\t{inter_list[i][-2]}\t{inter_list[i][-1]}\n'
            )

    with open(
        os.path.join(args.output_path, args.dataset, f"{args.dataset}.test.inter"), "w"
    ) as file:
        file.write(
            "user_id:token\titem_id_list:token_seq\tage:token\tgender:token\toccupation:token\n"
        )
        for i in range(train_len + valid_len, len(inter_list)):
            file.write(
                f'{inter_list[i][0]}\t{" ".join(inter_list[i][1:-3])}\t{inter_list[i][-3]}\t{inter_list[i][-2]}\t{inter_list[i][-1]}\n'
            )


if __name__ == "__main__":
    args = parse_args()

    # load interactions from raw rating file
    rating_inters = preprocess_rating(args)

    # load item text from raw meta data file
    item_text_list = preprocess_text(args, rating_inters)

    # user profile
    userprofile = preprocess_userprofile(args)

    # split train/valid/test
    train_inters, user2index, item2index = generate_training_data(
        args, rating_inters, userprofile
    )

    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))

    # generate PLM emb and save to file
    generate_item_embedding(
        args, item_text_list, item2index, plm_tokenizer, plm_model, word_drop_ratio=-1
    )
    # pre-stored word drop PLM embs
    if args.word_drop_ratio > 0:
        generate_item_embedding(
            args,
            item_text_list,
            item2index,
            plm_tokenizer,
            plm_model,
            word_drop_ratio=args.word_drop_ratio,
        )

    # save interaction sequences into atomic files
    convert_to_atomic_files(args, train_inters)
