import os
start = 175138
end = 175156

for i in range(start,end+1):
    os.system(f'bkill {i}')