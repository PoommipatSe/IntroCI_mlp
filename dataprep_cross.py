file = open('cross_dataset.txt', 'r')

i = 1
temp_line = ''
for line in file:
    if i%3 !=0:
        temp_line = temp_line + ' ' + line.replace('\n', ' ')
        print(temp_line)
    else:
        temp_line = temp_line + ' ' + line
    i += 1
file.close()



file_w = open('cross_dataset_cleaned.txt', 'w')

file_w.write(temp_line)
file_w.close()
print(i)