
def merge_select_stru_out(md_out,database_out,select_stru_index):
    string = 'start'
    with open(md_out,'r') as f:
        lines = f.readlines()
    line_index_list = []
    for line_index, line in enumerate(lines):
        if string in line:
            line_index_list.append(line_index)
    temp = []
    for a in select_stru_index:
        l_1 = line_index_list[a]
        l_2 =line_index_list[a+1]
        tt =lines[l_1:l_2]
        temp += tt
    with open(database_out, 'a') as f:
        f.writelines(temp)

if __name__ == '__main__':
    md_out = database_out = 'database.out'
    select_stru_index =[0]
    merge_select_stru_out(md_out, database_out, select_stru_index)


