import numpy as np
import os



def get_grouped_list_of_files(directory, output_path, group_tags=[""], prefixes=[""], write_files=True):

    files_out = {}
    for k, file in enumerate(os.listdir(directory)):
        if group_tags[0] in file and '.nc' in file:
            for l, tag in enumerate(group_tags):
                file_look_for = file.replace(group_tags[0], tag)
                if np.any(file_look_for == np.array(os.listdir(directory))):
                    if tag not in files_out.keys():
                        files_out[tag] = [file_look_for]
                    else:
                        files_out[tag].append(file_look_for)
    if write_files:
        k = 0 
        for key, value in files_out.items():
            with open(os.path.join(output_path, prefixes[k] + key + ".txt"), 'w') as fp:
                for line in value:
                    fp.write("%s\n" % os.path.join(directory, line))
    
    return files_out