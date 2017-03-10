Example to get all the records from HICSS

1. run getConf.py to generate the booktitle.txt which contains all the conference records
2. Check how many lines of HICSS records 
    grep -ni HICSS ./booktitle.txt | wc -l
    11882
3. Check the start and end line of the HICSS records in the booktitle.txt
    grep -ni HICSS ./booktitle.txt | head -1
    3055298:HICSS
    grep -ni HICSS ./booktitle.txt | tail -1
    3067179:HICSS
4. The result(11882 = 3067179 - 3055298) indicats the HICSS records fall into the region [3055298,3067179]. Run getAuthor.py, getTitle.py, getYear.py and Extract the HICSS records from title.txt, authors.txt, year.txt
    sed -n -e '3055298,3067179 p' title.txt > HICSS_titles.txt
    sed -n -e '3055298,3067179 p' authors.txt > HICSS_authors.txt
    sed -n -e '3055298,3067179 p' year.txt > HICSS_years.txt
  
    
