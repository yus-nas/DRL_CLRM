WELSPECS
--     WELL      GROUP     I0   J0 PDEPTH TYPE RE INFLOW SHUTIN CROSS
I1    Group   30   110     1*  WATER   /
P1    Group   1     1      1*  OIL   /
P2    Group   60    1      1*  OIL   /
P3    Group   1     220    1*  OIL   /
P4    Group   60    220    1*  OIL   /
/

COMPDAT
--    WELL    II JJ K1 K2 STAT KR_TAB  TRANS   DIAM     KH   SKIN     DF WELLDIR     r0 END
-- 
I1 30  110 1 1 OPEN 2* 0.30 1* 0 1* Z   /
P1 1   1   1 1 OPEN 2* 0.30 1* 0 1* Z   /
P2 60  1   1 1 OPEN 2* 0.30 1* 0 1* Z   /
P3 1   220 1 1 OPEN 2* 0.30 1* 0 1* Z   /
P4 60  220 1 1 OPEN 2* 0.30 1* 0 1* Z   /
/

WCONINJE
I1 WATER OPEN RATE 100 1* 450   /
/

WCONPROD
--     WELL     STATUS MODE  OIL WATER GAS LIQ RES BHP WHP
P1 OPEN LRAT 3* 20 2*   /
P2 OPEN LRAT 3* 20 2*   /
P3 OPEN LRAT 3* 20 2*   /
P4 OPEN LRAT 3* 20 2*   /
/


