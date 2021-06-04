# Training data extraction from GPT-2

This repository contains code for extracting training data from GPT-2, following the approach outlined in the following paper:

**Extracting Training Data from Large Language Models**<br>
*Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and Colin Raffel*<br>
USENIX Security Symposium, 2021<br>
https://arxiv.org/abs/2012.07805

**WARNING:**
*The experiments in our paper relied on different non-public codebases, and also involved a large amount of manual labor.
The code in this repository is thus not meant to exactly reproduce the paper's results, but instead to illustrate the paper's approach and to help others perform similar experiments.*<br>
*The code in this repository has not been tested at the scale considered in the paper (600,000 generated samples) and might find memorized content at a lower (or higher) rate!*

## Installation ##
You will need [transformers](https://github.com/huggingface/transformers), [pytorch](https://pytorch.org/) and [tqdm](https://pypi.org/project/tqdm/).
The code was tested with transformers v3.0.2 and torch v1.5.1.

## Extracting Data ##

Simply run
```bash
python3 extraction.py --N 1000 --batch-size 10
```
to generate 1000 samples with GPT-2 (XL). The samples are generated with top-k sampling (k=40) and an empty prompt.

The generated samples are ranked according to four *membership inference* metrics introduced in our paper:
- The log perplexity of the GPT-2 (XL) model.
- The ratio of the log perplexities of the GPT-2 (XL) model and the GPT-2 (S) model.
- The ratio of the log perplexities for the generated sample and the same sample in lower-case letters.
- The ratio of the log perplexity of GPT-2 (XL) and the sample's entropy estimated by Zlib.

The top 10 samples according to each metric are printed out. These samples are likely to contain verbatim text from the GPT-2 training data.

### Conditioning on Internet text

In our paper, we found that prompting GPT-2 with small snippets of text taken from the Web increased the chance of the model generating memorized content.

To reproduce this attack, first download a slice of the [Common Crawl](https://commoncrawl.org/) dataset:

```bash
./download_cc.sh
```

This will download a sample of the Crawl from May 2021 (~350 MB) to a file called `commoncrawl.warc.wet`.

Then, we can run the extraction attack with Internet prompts:

```bash
python3 extraction.py --N 1000 --internet-sampling --wet-file commoncrawl.warc.wet
```

### Sample outputs

Some interesting data that we extracted from GPT-2 can be found [here](Samples.md).<br>

Note that these were found among **600,000** generated samples. 
If you generate a much smaller number of samples (10,000 for example), you will be less likely to find memorized content.

## Citation

If this code is useful in your research, you are encouraged to cite our academic paper:
```
@inproceedings{carlini21extracting,
  author = {Carlini, Nicholas and Tramer, Florian and Wallace, Eric and Jagielski, Matthew and Herbert-Voss, Ariel and Lee, Katherine and Roberts, Adam and Brown, Tom and Song, Dawn and Erlingsson, Ulfar and Oprea, Alina and Raffel, Colin},
  title = {Extracting Training Data from Large Language Models},
  booktitle = {USENIX Security Symposium},
  year = {2021},
  howpublished = {arXiv preprint arXiv:2012.07805},
  url = {https://arxiv.org/abs/2012.07805}
}
```



