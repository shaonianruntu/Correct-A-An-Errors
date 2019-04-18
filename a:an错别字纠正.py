# encoding=utf-8
import kenlm
from collections import Counter
import nltk
import re
from itertools import product


input_file_name = "./test_set_public"
output_file_name = "./output_file.txt"


#按行读取文件，返回文件的行字符串列表
def read_file(file_name):
    fp = open(file_name, "r")
    content_lines = fp.readlines()
    fp.close()
    #去除行末的换行符
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")
    return content_lines


#对句子中的a/an进行全排列组合，返回全排列后的所有组合字符串列表
def change_a_an(line):
    new_lines = []
    if "a" in line or "an" in line:
        #获取a/an的总数量
        a_an_counter = Counter(nltk.word_tokenize(line))
        a_an_num = a_an_counter["a"] + a_an_counter["an"]
        #对字符串中的百分号进行出来，反正在后面的字符串处理中进行转制，从而导致字符串参数传入个数不匹配
        percentage_regex = re.compile(r"%")
        new_line = percentage_regex.sub(r"%%", line)   
        #对字符串行按照a/an进行切分
        a_and_an_regex = re.compile(r"""
            \sa\sa\s | 
            \sa\san\s |
            \san\sa\s |
            \san\san\s
            """, re.VERBOSE)
        new_line = a_and_an_regex.sub(r" %s %s ", new_line)
        a_an_regex = re.compile(r"\sa\s|\san\s")
        new_line = a_an_regex.sub(r" %s ", new_line)
        a_an_regex_front = re.compile(r"^a\s|^an\s")
        new_line = a_an_regex_front.sub(r"%s ", new_line)
        a_an_regex_quotatio = re.compile(r"([^a-zA-Z]'a\s)|([^a-zA-Z]'an\s)")
        new_line = a_an_regex_quotatio.sub(r"'%s ", new_line)  
        #长度为a_an_num的a/an的排列组合方式的枚举
        a_an_form = list(product(("a", "an"), repeat=a_an_num))
        #按照排列组合枚举对字符串列表进行组合，形成新的句子
        for form in a_an_form:
            new_lines += [new_line % form]
    return new_lines


#主函数
if __name__ == "__main__":
    lines = read_file(input_file_name)
    output_file = open(output_file_name, "w")
    
    wrong_line_num = 0

    model = kenlm.LanguageModel("./lm.bin")

    for i in range(len(lines)):
        line = lines[i]
        try:
            new_lines = change_a_an(line)
        except TypeError:
            continue

        #得分判断
        line_best = line
        line_best_score = model.score(line, bos=True, eos=True)
        for new_line in new_lines:
            if model.score(new_line, bos=True, eos=True) > line_best_score:
                line_best = new_line
                line_best_score = model.score(new_line, bos=True, eos=True)
        if line_best != line:
            output_file.write("%s. " % (i+1) + line + "\n")
            output_file.write("【Wrong~】")
            output_file.write("=> " + line_best + "\n\n")
            wrong_line_num += 1
        else:
            output_file.write("%s. " % (i+1) + line + "\n")
            output_file.write("【Correct!】\n\n")
     
  
    print("The total number of sentences that need to be corrected is: " + str(wrong_line_num))   
    output_file.close()
                


