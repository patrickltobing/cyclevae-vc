# awk '

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

BEGIN {
    min_idx=0;
    min_val15=9999.9;
    min_val17=9999.9;
} {
    if ($2=="average" && $3=="evaluation") {
        split($1,str1,")");
        split(str1[1],str2,":");
        idx=str2[2];
        printf "%d: %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;;\n", idx, tmp_trn1, tmp_trn2, tmp_trn3, tmp_trn4, tmp_trn5, tmp_trn6, tmp_trn7, tmp_trn8, tmp_trn9, tmp_trn10, tmp_trn11, tmp_trn12, tmp_trn13, tmp_trn14, tmp_trn15, tmp_trn16, tmp_trn17, tmp_trn18, tmp_trn19, tmp_trn20, $21, $22, $24, $33, $34, $37, $38, $41, $43, $44, $46, $47, $49, $58, $59, $62, $63, $66, $68, $69;
        if ($59+$63<=min_val15+min_val17) {
            min_idx=idx;
            min_val1=$21;
            min_val2=$22;
            min_val3=$24;
            min_val4=$33;
            min_val5=$34;
            min_val6=$37;
            min_val7=$38;
            min_val8=$41;
            min_val9=$43;
            min_val10=$44;
            min_val11=$46;
            min_val12=$47;
            min_val13=$49;
            min_val14=$58;
            min_val15=$59;
            min_val16=$62;
            min_val17=$63;
            min_val18=$66;
            min_val19=$68;
            min_val20=$69;
            min_trn1=tmp_trn1
            min_trn2=tmp_trn2
            min_trn3=tmp_trn3
            min_trn4=tmp_trn4
            min_trn5=tmp_trn5
            min_trn6=tmp_trn6
            min_trn7=tmp_trn7
            min_trn8=tmp_trn8
            min_trn9=tmp_trn9
            min_trn10=tmp_trn10
            min_trn11=tmp_trn11
            min_trn12=tmp_trn12
            min_trn13=tmp_trn13
            min_trn14=tmp_trn14
            min_trn15=tmp_trn15
            min_trn16=tmp_trn16
            min_trn17=tmp_trn17
            min_trn18=tmp_trn18
            min_trn19=tmp_trn19
            min_trn20=tmp_trn20
        }
    } else if ($2=="average" && $3=="optimization") {
        tmp_trn1=$21;
        tmp_trn2=$22;
        tmp_trn3=$24;
        tmp_trn4=$33;
        tmp_trn5=$34;
        tmp_trn6=$37;
        tmp_trn7=$38;
        tmp_trn8=$41;
        tmp_trn9=$43;
        tmp_trn10=$44;
        tmp_trn11=$46;
        tmp_trn12=$47;
        tmp_trn13=$49;
        tmp_trn14=$58;
        tmp_trn15=$59;
        tmp_trn16=$62;
        tmp_trn17=$63;
        tmp_trn18=$66;
        tmp_trn19=$68;
        tmp_trn20=$69;
    }
} END {
    printf "#min=%d: %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;;\n", min_idx, min_trn1, min_trn2, min_trn3, min_trn4, min_trn5, min_trn6, min_trn7, min_trn8, min_trn9, min_trn10, min_trn11, min_trn12, min_trn13, min_trn14, min_trn15, min_trn16, min_trn17, min_trn18, min_trn19, min_trn20, min_val1, min_val2, min_val3, min_val4, min_val5, min_val6, min_val7, min_val8, min_val9, min_val10, min_val11, min_val12, min_val13, min_val14, min_val15, min_val16, min_val17, min_val18, min_val19, min_val20;
}

