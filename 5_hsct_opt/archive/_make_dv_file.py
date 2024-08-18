import os

# read in the optimization history file and get the last DV line
# then make a DV file out of it..

hdl = open("CF-oneway_design.txt", "r")
lines = hdl.readlines()
hdl.close()

last_dv_line = None
for line in lines:
    if "Design" in line:
        # print(line)
        last_dv_line = line.strip()
        last_dv_line = "'" + last_dv_line.split("{")[1].split("}")[0]

# now we have the last DV line
# print(last_dv_line[:30])
# print(last_dv_line[-50:])
# exit()
dict_parts = last_dv_line.split(",")
# print(dict_parts)
dict_names = [chunk.split(":")[0][2:-1] for chunk in dict_parts]
dict_values = [float(chunk.split(":")[1].strip()) for chunk in dict_parts]
#print(dict_names[-10:])
#print(dict_values[:10])

# now write the new file
dv_file_hdl = open("CF-sizing.txt", "w")
dv_file_hdl.write("Discipline structural\n")
ndvs = len(dict_names)
for idv in range(ndvs):
    name = dict_names[idv]
    val = dict_values[idv]

    dv_file_hdl.write(f"\tvar {name} {val}\n")

dv_file_hdl.close()
