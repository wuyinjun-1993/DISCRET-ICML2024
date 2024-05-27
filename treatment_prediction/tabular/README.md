# Prepare data

Let us assume that the folder for including all the datasets is /path/to/data/.

For IHDP dataset, we need to download all the csv files from [here](https://github.com/claudiashi57/dragonnet/tree/master/dat/ihdp/csv) and put them in the folder "/path/to/data/ihdp/". For IHDP\_cont (or news\_cont) dataset, execute the following commands:
```bash
cp data/ihdp_cont/*.pt /path/to/data//ihdp_cont/
cp data/news_cont/*.pt /path/to/data//news_cont/
```
