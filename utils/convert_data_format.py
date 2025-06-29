from prompt_template import (
    second_round_litigants_extract_template_A,
    second_round_litigants_extract_template_B
)
from utils import *


def convert_testset_to_qwen3_fast_format(input_path, output_path):
    identity_lookup = {
        "上诉状": ["上诉人", "被上诉人", "原审原告", "原审被告", "第三人"],
        "二审庭审笔录": ["上诉人", "被上诉人", "原审原告", "原审被告", "第三人"],
        "民事判决书（一审）": ["原审原告", "原审被告", "原审第三人"],
    }

    prompt_lookup = {
        "上诉状": second_round_litigants_extract_template_A,
        "二审庭审笔录": second_round_litigants_extract_template_A,
        "民事判决书（一审）": second_round_litigants_extract_template_B,
    }

    ori_json = read_dict_json(input_path)
    out_ljson = []
    for _, juanzong_dict in ori_json.items():
        for wenshu_type, d_list in juanzong_dict.items():
            identity_list = identity_lookup[wenshu_type]
            prompt_template = prompt_lookup[wenshu_type]
            for single_d in d_list:
                ori_text = single_d["text"]
                ori_text = process_special_pos(wenshu_type, ori_text, "S", 2048)
                instruction = prompt_template.replace("{{ori_text}}", ori_text)
                output = convert_litigants_json_to_text(single_d["target_object"], identity_list)
                new_d = {
                    "instruction": instruction,
                    "input": "",
                    "output": output
                }
                out_ljson.append(new_d)
            
    write_dict_json(out_ljson, output_path, indent=2)


if __name__ == "__main__":
    input_path = "/home/workspace/Qwen3-32b/corpus/1.valid/anhao_to_annoed_data.json"
    output_path = "/home/workspace/Qwen3-32b/corpus/1.valid/anhao_to_annoed_data_qwen3_format.json"
    convert_testset_to_qwen3_fast_format(input_path, output_path)