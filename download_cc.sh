wget "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-21/segments/1620243988696.23/wet/CC-MAIN-20210505203909-20210505233909-00000.warc.wet.gz"
gzip -d CC-MAIN-20210505203909-20210505233909-00000.warc.wet.gz
mv CC-MAIN-20210505203909-20210505233909-00000.warc.wet commoncrawl.warc.wet
