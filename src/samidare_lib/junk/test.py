import numpy as np
import samidare_lib.decoder.binary_dumper_version3 as test

if __name__ == '__main__':

    print("check flag generator")

    boolflag = [True, False]

    for flag1 in boolflag:
        for flag2 in boolflag:
            for flag3 in boolflag:
                for flag4 in boolflag:
                    for flag5 in boolflag:
                        flag_binary, length = test.pack_many_inverted(flag1, flag2, flag3, flag4, flag5)  
                        re = test.unpack_inverted(flag_binary, length )
                        print(flag_binary, length, re)
