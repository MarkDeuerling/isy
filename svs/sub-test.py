#!/usr/bin/python
from re import sub


C="dingnb rnsr yebdn kir nivnb gewgrirerimvgthiaabn onbgthzenggnzr fwnb ynvv kfv din hfneaiugrn vfth wethgrfwnv pnvvrpfvvk fv dnv ivhfzr gthmv nbbfrnv"
print(C)

C = C.replace("n","E")
C = C.replace("v","N")
C = C.replace("i","I")
C = C.replace("g","S")
C = C.replace("b","R")
C = C.replace("f","A")
C = C.replace("r","T")
C = C.replace("d","D")
C = C.replace("s","X")
C = C.replace("o","V")
C = C.replace("t","C")
C = C.replace("h","H")
C = C.replace("z","L")
C = C.replace("e","U")
C = C.replace("w","B")
C = C.replace("y","W")
C = C.replace("k","M")
C = C.replace("m","O")
C = C.replace("a","F")
C = C.replace("u","G")



print(C)
print(sub("[a-z]", '*', C))
