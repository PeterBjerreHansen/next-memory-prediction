from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import tempfile
import zlib


_SHA256 = "f553f8cb51a7da97f94b8571d2e0342a18b722bc11bc7a2ca6b9ec9707399b21"
_ENCODED = """
c-qZfTbtv?aeeo%phmXkB#JV_3ogDy*6QM2X;+f9Yp-QX-x?%A5+Vpt05RN=CBG-XLO(`7Ea+31L7gfl&v{vWT?00%yQ}M*s_KUR@w3U~>NaoMqOPufJ-NDh
=lbgHAUic(waQK+uDY^>^q8&JMYYwUUXrh0o$7sFwY~gry=*f12mbB<TC987_1pK&Rb8F(>J;Ae+wc3%zW>jwhrhLb$7TNNTsPr2o2+c}hrgBWsVR<5+*}q{
b#utd;=9~;)E@Iyk(Ir?A7BpY4{u*k`R209dENfMP`Us2Ys~(`?;2XBCYKc%{vcz!I^7-fFplT>w_RSX@?jw1@rS3!H5{iz{q)h@DSw*Z=H<gAhzY#v``8q(
U$w_<l@F?_C;BlTUe%jTo1a?$*RIW9HTgDwt;hJVFc<aspx73DSE`euS{Jv)y35K}YxyI0hW~#3v!UCoW8I#5rR!DRwlz)OI$zc6OVeh@{(q1Ae;@Ou@rqK1
{>9~m7k~cY{VUf^eXP5a(l_*1?Pa-H94xn}y52@#wcRop>7Z@s^UXK!yUeR~uO^JRe{LgxSKVxay8SEIFt3*VY`p3P<uY5@zunfWZ22&b$++3IFSy@Ldb7RJ
x26xRude#$?4kK)-<m(Pe%`kh53OJHt>~fk%f1yqwBG7l$wTW`eJg!v{Y!ZLsMEiO=BUoUh32Tw+o3t?^Y5WKD)dfhjvD<(XpSmfhvul$O=ylvO+$0kY8Iwp
)M_4@qgIR19JPu<bJQvh%~7i)G)Jw{&>Xe;b!d)S{U$U=t$rJtqgKBQt1)U7R&F+G^=@d6TKzsWN3Grq%~7jIp*d>xerS$beGr<XRv(7usMSZIIcoKXU=^cQ
k3)0R>PcvhT74XvqgJ1U=BU-v&>XdT7Mi10pN8hB)n}nOYV~<&j#~XOSm&tK7oj<7^*l63tzLxYsMViBbJXh3p*d>xGBii6{t}v_R$qqZsMTLXbJXf@VV8_r
ef3pnk7|7#nxk5;LUUB>@1Z%Wm4)W0R@m;*sMabpN3{Z~MWb55t3{()o3LF+wF13}N3C|DIcime=BU*-p*d<5JW)Jq6`W~2YIO+BQLErD<58<RG)JuhNGGFK
--hO>RltX2)GB!RWYj7+=w#HY3(Zlh+t3`fI)~<{RdBz_sMTF)j#_;e+}Ehp_n|pz_0P~8wfgD5LVHx}r~eM^QLmqV4DC^|H{T4JrjZhal}lqK2&<DON)Q$#
O_d;rPrX)>8j>3&IY2U1k`@x>?cZ!5nJdW=60L6riB|XoiPpG;M5}xUiPrfM9KY6i2Z`2s1&P-A42jmcf<){5J|tS_=a6WfZz0h-KY>K+{1Ota^9MjDt@9od
t@9_4Xq}gkXq}%yqILcj60P$F60P$KNVLwML!x#510-7KF9D%i=g%O~I==^r*7+$UTIY`-(K>$$iPrf$NVLu$L!x#55E8BP7m#S3Ujyg0&LAc?bFDO(!(1y3
J}}oxe*=kD`tOivrT+?vR{BqnXr&)RqLqFF60P*F!A7*we}qKq4Dz$kI{z6Gt@B%uXq|r!iPrfQBwFWpA<;U6J1w-%e}P2n{3}o<t+S9Qtushhq;>u$BwFXc
L85gA(%nQ_>EA%2mHs^>TIt_HqLqFV60P(vLF}~B@Ul0t)*6oXCe~`hH{Zls@83b9_5KAUT5V9$SgZX#B(ZYpaDlN_dk3Ga!0o3W;W-t){q!R|u!6XseuRHZ
wAK}ZV}*46+obbE>z&JE3-uWMjzYcuhk%7dYn~1LCrSoKoM_eIj8d&SoKdQ0ynrXxSoq@^Lrw`|pn9rRFED!qiggE()*YCVYSp25suc%bq+0PKc)hlj><%J5
=?6&kq$3!eUe9POEo2(YUtl4ogpalKrd1!F(so`*XIl9R5^d-iv>=Qu&^v)7@<r1dJ@0J>k)C(tSf+%NWeJfsnVtMlE5C(A>z<t;(z-u~BqeLW4+$)5{E*NR
5%iScG82(jzQhC)U?PJsCB!U5q<3nD=_1U`MG39_V@R~wz$B-$Yua=Wk;P+(vng8y>}#e?2Rma**jZriXL{}danqTe`=dOijs7vFg7C9GL!@W_Rt%URv?(Cc
13+41I@beOqW`&8zk_5>ddV)DFUaV`#t2FanYKCY+u_H03C{Nq>QNlAdlz~YYDo0vMPCG|wLGK^Xb<-|UFbnzh8AS~GHZmZ$T3Y9TK^}Ikon6FUuYA0AzPKe
wS*RtR(~g+AfoMyWr=tI5NZ8ILJnUHk|4H{zVr@2@@PsR+uIA~G14QrgBP9>(u!(?v=tN~q`ei9HloiUA;bdmOk+I)@ntbN&rHl9)<fvzIlTr?AxX*LhgT5h
Zbq>lMJC>mptmS7DHeN0sF&i5@h0p^<sEt`l{i4c-x9l$;J3qgla$y5ERzuUoXOgz+MMbQMC4DQ2!U{Ml3(gEJc2~~(-xy55ME<z5ekR(!Hf_RKZ8VD(_msV
0%J+@&lDQJ)k45%c7}|XA(LN{MagfONMO&+2#*iI=-IU%#s+{gBSda@5T!hG0HfK>H7(+UST9CfLqucQ$z$VK-jSRuA#-t%r}RixP==tn0=&&`^f)T{r5?uV
zB|1fdn6ADojYuc+4P3Z9gJ-@)uVVSJ7YGTu{ODMpv`KDIhYbqK<jBl8)$fo9?A<@CI!`R$XpRrhixG5CAeM!+h&B<^&p!%cL2koaQAlTa~zA1y%`0(JL!-R
uaO_d<d_Gc_PMA#%m}tobDGVsiJwD6t_ol<BiKFx^k#(HPoNw{zqMEb#lPJaA_9$-JVuOlEiyvwVgJrkLQ#x@VEYW?CEWgOc*#N!17J8?=uwDco)L8SuQ)<T
<B;yOSs@2b*j-|p2)s*~I*N-=@VB$*n%pwbX-2`Z_%w=+Ycb_W4`S2FPid3OcB2q^3wAn-IB1i1#58&G4$2D3L{X@m$$H0RQJCXcuZ8$hg77ATNDty0Nc1|a
!T4se8G@kIj9^?Qg+W}NB3D(bpKGH67R?e0<iKNQ6h5zHBnfQ|d38bo1*Vh&=(@&O5{f8R5K*#I`~qS4y2chF5MSLR55>_d+0H2k7C9o98Aa2XcnpQ)?~3yy
AkP}i0fo~!luotB-3~8^42PGo5S%ZE{YGfM1DBl<p7(v0C<N&>IfL5IDY|ZIh^9nm){trI+R9^kBnRnCk0irl&J~=$b%KC4xjdssAg+IYO-YRgl`FC-glh8}
1Azw!^$0o<&1g{aTRnp8HAEaHVeACtOKkBuMcEx@mw~*=F)|9Y>tVmlr`N>Fz`^GP<_B2coS?j{1?lF5<tx0LfV_8;`{79@SX;;<;!;8-<^<>M5h5a3r%Y7W
%kmtKd`>Crx)K?&s})3KSCVy@Q}jJl%jYvanrE^y1m^V_(?e)J*xQUL`(A|F`)*{t2+mu<%sB!268m$m5M7-4obdc0B0Y|yZ1y<=x(t{C@s&&}$KhL0l{rCr
jTIq84`aj9Q6Qd|_%TEJ;Na#9HZhqSLiI9(hyYPUdMM4k)lf{nxqo*=`e3pN*<V6J)+M`zqVh-L2ME|7$;PFy{5h!goPqt2%a4hkw8Ml^Y+mJO$f$6Jm5MkZ
goxIp1JunK+E*e=&E63(np1?{TdW8f7sa4ry(|x&BWAt@(VX|VSPubxQ4TNCE7P_RQHB}gCYW#Ur;;#!D-b;A6!}2_X-+u5gx(29iSVHW^F6nh9_Q5Kz2=1V
ne0^p``(>kJPEDP5+c1mM+EkBg8DN8$~i&(MmXP`^XAwJlsCsYr8yz}38SNkJsZ%Lp#CTZPEfysnGwt%hT#&-pCzcCr;L&a!xq=qb^!-(ySQewytkGGXVfdq
`GPQiC-KPQ+9nt8d)6b%hYZ4kiqAbNK^9X;SVE?ks9Qm#M|S9NMua)#4l+HWk0GIqN>f3mhw@x(U~xk!#!Md6!}ti2h&K{s6m5w}k3bN8K~a3R6~?k4@Gmof
@?v^pmJC8FE+KcknDU~C3tCXVeT~(lIDRdFxnS(a)D!TZhBad7Uk#3GLDlO0u=0fd6_%aAAE$08kUusM%{X#uAtWA?56ef`-vQ?rGiGJ4A)|HKK*XtQIH<);
FU*(7oh~>551+T7SiS%^Sx^}p2l*CM$Ig!s&1s_z-f2PL56--pN5q*2WL;1i{;oy<wjlgpVkiXub%)<BIN>F%aY5DXP6k8C_yS3^g#!P#5yvbjpx<Ej3HV!t
%nOR=Pr|VmROd!KyI|ypRTJ=^ho2Ji%WkKDzC&+>`^Te<m2m$V-1dS{Ulxu*AE!7Mg!&TXF9`O#71oVVzdK;n3H3|yG79luU<{OzFIstM%3%Uzv_A%XToBr4
c&mbXsbnlDobS;V6GceBAKVFH{dUlpl8<1#3kv58Au|guhwp00DD;xiQC7Z>Y%qTm(N}V&ctLsjqLN?J23bmYxS**1IP6yn>+j!_QfdQ=6Vk82@)wkw{{Rx&
9OBaTd9g4zm_&-{<*4(5kp3WvrUe}ytHpj(t~JYZdK||>^yB0NI*hnnULsc#(OI&rgNRbXV!;vNJA5r>CnAX70iPnu)PE~|QdT&VJX}Pj^Ub}G##2L-a-b!w
H@e~Y3-%Kc#@|9hPFO%NB9Pz81QEs~h>9pxf0oS=5z2RsJQy3GmRv}5W3m-P>=+T;=dW?flfholH6pOj%6lCs*nb4>8WG&LJ8asBg8klnk=lz0@S7S`Fe1PY
(>%--)!C&w6%pt^$0`xv!}CX!y2qZ62=1YeOca5B4#FClYWqn#;4>!j9zC3I#KA@c`tMcp7Rrf<^Na}gkJ2H9$+*iAQMmt1rc8nUa6}-Ycz=zoP(%SgBI$_2
{cbIfam`eo8%HseXCi|8O(u3va9>Ge8BqfNL!2|_6eSY#5utrmLBw&(YH;HbNBcQy0uk5WJLzCS#tixwDb)WmmsfL;3=gp2)Nv^<rXU$xgD@X~eZ-jGfILQ2
gfFW5El7p<4rDJP+&{^$nZ4Zi$%RLVEXp9Zi#X!1M8-#IFq)VHF}y{=|4W!E1^OpR%0zVZ4LR$G()qcBKM|GWw-R?m4F8w}0{=rd{5YY5JMb+8{!I;<7!mqc
6;8Jk_-`AL@zDs3H{pN<FG;w9iWo2=?9ZwPG3WT?J#^+;teCLB$_HV}g=m;@N@eZSfKdwnCG!-eT(yJdDHk54Kf)O4OpoP#Nc5U?1K)`h_)7^jN;&z`AdZd*
{AKsVg#FtbBBqwPJV&)6oa3jH<Xv-uj7Yx30x?DPt5ROVxL@vv*TnRwyochI9Ldg@iu4=2n*)0wR7?lw#39BE`*8m;q5ke($Y%{&$T*0^{RT?y?~CEbl&chF
=nPsR{gG|qF_YKtqY<H0F(Lk@ge}EWuCp}K1vz6$v&Ed<mzW`*GC&T($ZaAhOHuv~hmT^y`*pF#j8J~R8kUSOUtDcWpnnFq#{~H8{thw@V?@TqRq2mUP{_Oy
-f!e~OHAgBTPu{_$4R`H&)7+-A||+h5Ak<A*W&;sj|uMk;bst0q2k_c#1!Z+J9!NiS!Ku+=F0`lm_UD*OC2#L)IY&e_^2J?m6(Hk#6K~CekL|aA^rxpiRipt
wv=s6fZy&Uz=#*7T9HffB&Jv&ZZKA`|6(V>bWE_n0o92K^^v5B3H93(d}vI#KSWS5;eH?CJ;-U>2$scM!_VZt(}EZzuq;-{FC;J~;78mZM|=`NmXP!O!_qM9
Kip7`Ir7I5g_zKPB`>GQzslu=XiWIO#8gs}zr<7$_OCN}2W^==lb3K}Ok5wK|FXjsFv^O<trPUGWgG<kJyOV<`8=M?0+;dyd<gru7%t&{1C{wSUJbGn6Yk4#
!I%!@)kiUKJ`*oFrkK+EOPOfS@81p&at;IgJmoDTL@-YIY@uxGl#8xnw3Oz5F5D*OQ}RLwW5$0mh=d40A8tU=gromGM09Fcg3yGb|5aT=MpRiaG9eDIf$=2Q
oM1v^m|XMWgf(Q8ks0y<2?hVhd|07`X}}tn`zZE5Hj?2?hzRr>Od^?txBz(lK!Os=1AI7?ClWqTUu<^}Ms`|Dd3YS&Q@GoNSiw$sR6?|1h0Bdoz80{A?1oc3
!lDw&0c`KH!3kIVr96;O(!aMeS%0DhT_sOaPIG*ePsY@8ZBt|E>v3+081((BKM_5s?@3QW3}ICc1E2Ac23b!!&$pE4#?eWJKB0`j3P=1BA_!X;u_6d>!B`bL
cu%gP5(42F%KjH1*a<O%tt0>wPXFU>Kr-jcC$JXc1^xDt$)#+-vh1YG$OI{aEf76uPg1)`h#m~skWf<KBVk1eF@#;Wl|U(>LjW5PyoBojxZF!wft6Ht6Cw$Z
<=sRPde4Y-WkQDlGK41y9RjGqMUn-l(ZR_Qq6nYhLNKk*cM#o}x;KP+Ag{3@i5i4Nl1u}o1zrjqB%B!74d*5j$_yN3$|BC#93+vGP->tSw@M^ok8EAS9N{2^
#e^6_Co>dtm;ehVhHxw;=bTV&U@vw^34$jWWNZUC2<h4ZCXQ<ZjU@0Aq6kt&NGLymBUlM>g!a@(wkV<UKq-SEl7QTD!lwXEGL(cp8?>R~K#Oc*qG-aWB1jDQ
w(?s#eu<brNx9B(k_+Z~RZdcWOo%0X4hhj}@w`M4HVra42{D8vU_T*(utb=iP-&nB&q|0M^phYSDCKOr3?}7s0GQeIn%6~&nknT70K6&jgJqpbk9>0GwFrrY
<`TcBRQj(Z086j!bp+Wp>2+cPTx`em+H9N_7y5}M3{~iq&jDgiQ(_54F2B~p`V<m-Tp9i$rF20dVNXi=0=eyyQn~<h|L}BTlJ`T;dQ@vsl`;nflHDoYzUwNm
%XB(5p^`}XL|~1f@+CZZZow%n%rYPE--#<qryMk3K*SA}@)+d{<%mN{grFxk!=6d$h<_t_nUszN9$`_GGw7sCihlB8)o1S1iKN)m86RjqKt?qWnK-Hw901ZO
(E~YWlbX{3cn6&hK-nQR2Lt7&7D&Yk=Ir9?p>&<jIiq`!SB=AYBAwE)E&1U%l=l&RrHU2cE@Dbo-g|h#6c8WCD#=%*#0a)P`jjX^;CT>odR-0{q<lW0Mw&0B
q`}<)_9?M~eU9@JDJ2fhHAGxohYwHrjDV<4rvsNVtBMy4Sf3Ivcs>+QQ=$dJ^HX94=M2GUN}ND)Eh%w=!*Dn!rNqG}1E{1#3VIL#y-bM~pgxciC%Bbs9BFJ0
ZpbL9YJe0$N?C(*c3MfFbT~-f%$LW&_fld7xFeNP(O`&VQz8XD1jt}Gbs#4}Q@#Wc7#3E8@&{Qd?@CNXR~9be!%i}EA_uKJM=`f>q?B?9ArmW=z%=2ko4j6+
>I;0yh3J9!FJcE<*?DwyV2LZrDbWK-x}`)9s#?nbDP8T$miNEst4m`4iXyxmTvf^!jxp(Js=wxUb;z4-u0LkizkL$E4x^doBz&yqZF(Rx?bfu<x`S#GK9Hkl
b%Jj#XsT!*nS@WUn4i>IkWKKJ1^&s-v^S>xLcg;JJ?IC{Jh0xq^T_2VG&l7md<=+RQku3^a%I{_-C-Ekoq6D3XR0BsGduq;-1OGmi_Dc9nO(%nD%RLv9PCSI
K3hAV1<yu5Rd%{|7i?>5)i{|4p3^%j`;*!(T7GG4mDS5lpc{Q-`)~_oHGOQ_#mQ|8`;zSyCo3=5M^V`ySrh!~N(+wmiN}|QOI!GMg70gXy-{dEXUDs|;5TTm
9K$qcG@3R{V?j1job1M4yPusW^3?*(A9-it?X_v&*)i8zu%F-yE&LPOdb`|O!Ro@8pXl~;v-4!P76)s6Yr89o)^4qnQ`WpVx2sWXty?}><@Z<og#l}BcN2W*
MUw&BO=Yd=_JVGoxeaM&x+Y(;3x;t#T7#?1kDii=hOLlu$D7IhHwMh!ElV=G(hmIOx8B)~3rf=K!d%p}D@xuseN<y1V0YQYSJX|-=q6+E`^Gw;M|S0MyRqsk
6PRCcIL}NQS#-KLz^{JGf(y3!nWuMxkJK2eXlaLrsVZrH4-<U6$1HH&*uBDT^%SP2%-yo!g8=52wd+bf@{aMXVJBPbUs?ARCMP>sNp6uKd;ox_dG98JZCgyV
p`SgS$R{!Q&cE|)%S$6A9nZe+7iRL`a7YvG`CoijcXDGPQ<WS3)&OozU!LgftTQSayCvZWo3*vH(m4djHtiF8UbEWH;S<LeopE%QtOQe7F*93R>ww>3cd*N?
#%F7GpH1W&0fsxYc9<m-vb?nZsk`9L-?MwEb^^lIli?HLroQp{LNOdfvk8XX-Z}ci-hDg42c+y$-MPtegqSfQJJ^9c=AGFUZ!YlJZ>-B@zO&6GJZdcSaE8;k
F%1{-%|6bp;jZ@jk(Keh;(qvbrCW;H;SGm2!y41pTi2G*t|r6hUaUTqcWedoZL_i8b*>ms?Cn0M=xb}bm5Xc_Jyuq_K&2oDxrM7Ci#a=@+E|jgbCK`XK60?e
nwjN!%X-i9!VaywljA|wE{Y4Yey~<|aOyk9_}XDTPiJkdsk7ee)Z{KExv(ujxNdXH2-tecdLo~OF*a86L-j;HIc9Nb>)1udu-n>5qYav7jivn^566LiYc~*V
tlBzD<n8-xf=?Dus^?<x&U~jZ0M^sV2|i$82x2(oM_E+X(mGCAfZT?g;TpUhymJH@N7!K_>|D0xcwrO_=B3>Z=L=2-X0mq_Ftdy8@<Dd>!$aY^yQLF01b<0O
o!d!zWJr-n^2!=fcJPFPewMX%N<w|k#`Bq-|H^tD*naCEkqJb&&pwgwt60bD_gm%&ZEevivmUhWycHle<2SaKfOzYu{DG$mp0?(_eDJP*X`!|?Zv8&e^gZFy
Ivr(l8O3Vr!W_<hN-CrF7Ts7gEGh@cWY0OfW<h9&EN4cnqchBvrFWP!b`$5FBNdy*#cH^bJ{dj%VuL}sl5V5Q+$Bb8Hmd=^O^^A7C{;YZi?d4$Oe;Iq&haUm
cmCLI2e*-ryrPYRtx5QNw<V-*KRRy5XqW}-${5$n@K|?gci2FrKD){8*d?qjwj+gbuB>S?xcixJ#sBke+1qs#t!y^pOKYx%7vEXuQ05c)9)RW3xnn7>?R;-s
+L4{bmo7in<Sr&P6kHZ=X(8n3wgfJZ8)Nj$&?3}b?Sh4+_p^2(0j;;ZAe~dZFv*p-T?$}xiRN?{h^8YfF5!80@h#)y$|ia`>spb`ViywVQ<2$8q{YZ8Ypyn=
40pjDr#(2fV*L0EmzZ+7)Y>J#Pm|#jDQ32g+%Lx_ZRR#X*jyEF@U6p{y$_3hR0m&QyLqRM4q0XA2k9M#|Dv+gEnE7~1zX&Z9_JT41;OiV?d=1*FWG(WFEgi=
BfK?-%eAw-qP;|mi<RfV?9odXd1tna5>kBOBDcN6eGx4?a;M+gzJP!yuJmlINOZ3`EV#h%5G%Ac1$|3{u`E-pkUep2Gf-!1&fyxD+ijU}N1lV;mL|hTzUU6D
#re`>R8~%gYvP8b*Ho!)cMdt%n4N+YX=+D@hDkIKo4HN#-0nB$V|R9aa2MMdMD9phQJrk(vg7qb-%R8?7yAB;OGYn+-&?EUs()j>(lawD7Yd%J{@G&D-X+J@
yzD00zK|B*WX@WviG0S3-}Rx#<<@ePimeOFf8xpCAO<Y0(_&}@WM~Q$_(krkB0d#Zx*Wycf*hxI8b_Vrj(MvLshZLul0*1#mD)PVD|3f>f=^`G;=Pyb9amL-
zr2Ky+1}N&T|AI6Z3k1nbyS6hdgu(>50u<86YOlJNmXTIF?hMmWsMrQi_fmQl$8$pa7EgN9GOdk`=mg=_C+`%VzNlu`}<7A>GE*_8_m=<qy4@ugW@tY(HI*q
hH*7_792{S@Aufdc9|T9A1*^0E<f9>A;-A@1(R-Bl^z~ldTV2;Ugq*G)*%fCRn{&8%5i_*UIfR=61#$XdTHJKV^<zJ`GU7F>dLsb(pm0>(&WacUr>c*9}KW*
4ciO$ZlPmm4mjW@vmJMpyVR`Dah}~Fqjv9(0~kXbJe9Bf<}O=UI72+xY)@%7lSyX;xINt2?L*VikIDKR9LPBtDz|i`aO?IMC_wA#jalPP^_}fVm~odbZpuz}
);8{B!_sX#aIB2YDcnh($TwZhF_=p8iHUsQ%jPHCnW~NR&2?j|GBs<ERe_csYH#M@8nVsIm5x7EYPFs@=l8<mTv@wiY5kOkTgMZ*^lUV!UGU5=tcIkSwU#?u
7~D8|R=ETSA9y%>_p`L2$i{J4ud`%wDu@1@+i<RZYl~lvJ?${mj4WkfM_V7AVZQLGjHL^6WnL?nLftwO4;P8;>Bwg7kGtfqOu*b?<*OaFU0hsoHQQx4=iqpK
%?0GzW%wrYVGUbMIk>{$#vM{)ZjWbd^cf})SA}hAoio=`uAwE9&0v@~?gvkX@1*G4!7gcxyTLBY6mA9^AM(`IWe|KfE|K0jdPl+Ju~>P2vAvY#Tlo}jZz_(5
b*^5wo!s9Uwv~Ezcp*d{Zu;{5-^qA8m~Mq!AO>JY;rg!)ad?%MJPkIdI%N9XC`vAgD`R`_!s%u{b6J#^>hQ&_BV<MCsKOm@hSG7}WogR4a=X{BEO5qTlrMPU
k0$cL6Ta0ukuM<GJ?D-sADw3?{b^th-cUv;s5qT%Eg7sGaw7$Lu+aL6Rp`456+dfv%n7@LPvpMz31Laan9A?bF~CiBu<3xII#fE&e6hx7<KxZ7QQy`deCA_b
dv{jkiMyX5!G@)n3~h~1>Qs)6=N9kUQ)4p?LuhNa&9|LD>fo3tA5LpcPU%yN?BVjz#aM@r3Cg+6J6pl*8k?c{+~v=5Ux{?*Di|`CE<~&y|LCfd^9T~^ae&-j
p0|RN1P;JZfig=8Z+mj9l(nvSXF`S{R5MAhaKqMN#J+GRFF^xrmDbGJP>{3+^Qk#bcyz&d;c2@)?LFM5oeURn&28F?xqe>G;^2-kB2UM+4^Fi^@V){SbOm}2
2JMNPp(tWpL*vewx+@ZmFV_kMICYKht8Hz#b#gn=*`}?G2Zl3||62#$l(h-M%uUU~0o~3YfOO}$Tt&;DVOdGp!(t=q07uV-sO<2>m5><N-!~D1yZh2@Hm<CS
<4PRvZyibwjE9f0P2>yPl*9t1wU;Vz36}1^PVfnFd+NhQDXrs_%vhIZCNJRR^U3hdU{h7%RRPy&2sSd85jwa-0zduu$z=F~0^f6W5!Oo!{I~X?^L@cNcXW|2
W<al%yEK$Lid(w#ZmzImqAPK?x4Az9R$kfrtakrN(yI1&%`w;7S87_@)yCH$!yVR%e9)F|uwJ<M^o5(!*4lRFg4p6@B^>)!wZu>^ce5<~k$i2nFa6yQjHJAR
;OqY1AAa`3{{T*!RsH
"""


def tokenizer_bytes() -> bytes:
    encoded = "".join(_ENCODED.split()).encode("ascii")
    payload = zlib.decompress(base64.b85decode(encoded))
    if hashlib.sha256(payload).hexdigest() != _SHA256:
        raise RuntimeError("vendored TinyStories tokenizer checksum mismatch")
    return payload


def materialize_tokenizer() -> Path:
    root = Path(tempfile.gettempdir()) / "nmp-assets"
    path = root / f"tinystories-tokenizer-{_SHA256[:12]}.json"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != _SHA256:
        root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(tokenizer_bytes())
        temporary.replace(path)
    return path

