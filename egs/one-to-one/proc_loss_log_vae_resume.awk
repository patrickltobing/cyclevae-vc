# awk '

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

BEGIN {
    min_idx=0;
    min_val15=9999.9;
    min_val17=9999.9;
} {
    if ($6=="average" && $7=="evaluation") {
        split($5,str1,")");
        split(str1[1],str2,":");
        idx=str2[2];
        printf "%d: %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;;\n", idx, tmp_trn1, tmp_trn2, tmp_trn3, tmp_trn4, tmp_trn5, tmp_trn6, tmp_trn7, tmp_trn8, tmp_trn9, tmp_trn10, tmp_trn11, tmp_trn12, tmp_trn13, tmp_trn14, tmp_trn15, tmp_trn16, tmp_trn17, tmp_trn18, tmp_trn19, tmp_trn20, $25, $26, $28, $37, $38, $41, $42, $45, $47, $48, $50, $51, $53, $62, $63, $66, $67, $70, $72, $73;
        if ($63+$67<=min_val15+min_val17) {
            min_idx=idx;
            min_val1=$25;
            min_val2=$26;
            min_val3=$28;
            min_val4=$37;
            min_val5=$38;
            min_val6=$41;
            min_val7=$42;
            min_val8=$45;
            min_val9=$47;
            min_val10=$48;
            min_val11=$50;
            min_val12=$51;
            min_val13=$53;
            min_val14=$62;
            min_val15=$63;
            min_val16=$66;
            min_val17=$67;
            min_val18=$70;
            min_val19=$72;
            min_val20=$73;
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
    } else if ($6=="average" && $7=="optimization") {
        tmp_trn1=$25;
        tmp_trn2=$26;
        tmp_trn3=$28;
        tmp_trn4=$37;
        tmp_trn5=$38;
        tmp_trn6=$41;
        tmp_trn7=$42;
        tmp_trn8=$45;
        tmp_trn9=$47;
        tmp_trn10=$48;
        tmp_trn11=$50;
        tmp_trn12=$51;
        tmp_trn13=$53;
        tmp_trn14=$62;
        tmp_trn15=$63;
        tmp_trn16=$66;
        tmp_trn17=$67;
        tmp_trn18=$70;
        tmp_trn19=$72;
        tmp_trn20=$73;
    }
} END {
    printf "#min=%d: %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ; %.3f %.3f dB %.3f dB , %.3f %.3f dB (+- %s %.3f dB (+- %s , %.3f %.3f ;;\n", min_idx, min_trn1, min_trn2, min_trn3, min_trn4, min_trn5, min_trn6, min_trn7, min_trn8, min_trn9, min_trn10, min_trn11, min_trn12, min_trn13, min_trn14, min_trn15, min_trn16, min_trn17, min_trn18, min_trn19, min_trn20, min_val1, min_val2, min_val3, min_val4, min_val5, min_val6, min_val7, min_val8, min_val9, min_val10, min_val11, min_val12, min_val13, min_val14, min_val15, min_val16, min_val17, min_val18, min_val19, min_val20;
}

