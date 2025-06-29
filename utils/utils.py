import json

def convert_litigants_json_to_text(target_dict: dict, identity_list: list) -> str:
    sub_identity_list = ["合伙人", "负责人", "法定代理人", "法定代表人", "委托诉讼代理人"]
    litigant_text_list = []

    for identity in identity_list:
        for litigant_d in target_dict.get(identity, []):
            starting_pair = ""
            str_pairs = ""
            sub_litigant_text_list = []

            for k1, v1 in litigant_d.items():
                if isinstance(v1, str):
                    if "姓名" in k1 or "名称" in k1:
                        starting_pair += f"{k1}:{v1}|"
                    else:
                        k1_abbr = k1.replace(identity, "")
                        str_pairs += f"{k1_abbr}:{v1}|"
                elif isinstance(v1, list):
                    for sub_litigant_d in v1:
                        sub_starting_pair = ""
                        sub_str_pairs = ""
                        for k2, v2 in sub_litigant_d.items():
                            if "姓名" in k2 or "名称" in k2:
                                k2_abbr = k2.replace(identity, "")
                                sub_starting_pair += f"{k2_abbr}:{v2}|"
                            else:
                                for sub_identity in sub_identity_list:
                                    if sub_identity in k2:
                                        if "关系" in k2:
                                            k2_abbr_idx = k2.index("与")
                                            k2_abbr = k2[k2_abbr_idx:]
                                        else:
                                            k2_abbr = k2.replace(identity, "").replace(sub_identity, "")
                                        sub_str_pairs += f"{k2_abbr}:{v2}|"
                        sub_text = sub_starting_pair + sub_str_pairs
                        sub_text = sub_text.rstrip("|")
                        sub_litigant_text_list.append(sub_text)


            litigant_text = starting_pair + str_pairs
            litigant_text = litigant_text.rstrip("|") + "\n" + "\n".join(sub_litigant_text_list)
            litigant_text = litigant_text.rstrip("\n")
            litigant_text_list.append(litigant_text)

    target_text = "\n\n".join(litigant_text_list)
    return target_text


def read_ljson(path):
    data = []
    for line in open(path, "r", encoding="utf-8"):
        if line and line != "\n":
            data.append(json.loads(line))
    return data


def write_ljson(data, path, mode="w"):
    with open(path, mode, encoding="utf-8") as file:
        for d in data:
            file.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_dict_json(path) -> dict:
    return json.load(open(path, "r", encoding="utf-8"))


def write_dict_json(data, path, indent=None):
    json.dump(data, open(path, "w+", encoding="utf-8"), ensure_ascii=False, indent=indent)


def process_special_pos(overhead: str, content_list: list, special_pos, segment_length) -> str:
    ori_input_str = "".join(content_list)
    if "起诉状" in overhead or "上诉状" in overhead:
        for keyword in ["上诉请求", "诉讼请求", "诉请", "事实和理由", "事实与理由", "事实及理由"]:
            idx = ori_input_str.find(keyword)
            if idx != -1 and special_pos == "S":
                segment_length = min(segment_length, idx)
    if "庭审笔录" in overhead:
        for keyword in ["异议", "上诉请求", "诉讼请求", "诉请", "事实和理由", "事实与理由", "事实及理由", "答辩意见"]:
            idx = ori_input_str.find(keyword)
            if idx != -1 and special_pos == "S":
                segment_length = min(segment_length, idx)
    if "民事判决书" in overhead:
        for keyword in ["上诉请求", "诉讼请求", "诉请", "审理终结"]:
            idx = ori_input_str.find(keyword)
            if idx != -1 and special_pos == "S":
                segment_length = min(segment_length, idx)

    if ori_input_str:
        if special_pos == "S":
            output_str = ori_input_str[:segment_length]
        elif special_pos == "E":
            output_str = ori_input_str[segment_length:]
        else:
            output_str = ori_input_str
        return overhead + output_str
    else:
        return ""