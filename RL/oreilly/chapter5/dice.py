##ヒャッハー！新しい章の始まりだぜぇぇぇ！！！
#####さいころを２個降った後の期待値

p=1/36
ps={2:p,3:2*p,4:3*p,5:4*p,6:5*p,7:6*p,8:5*p,9:4*p,10:3*p,11:2*p,12:p}

V=0
for x,p in ps.items():
    V+=x*p
print(V)