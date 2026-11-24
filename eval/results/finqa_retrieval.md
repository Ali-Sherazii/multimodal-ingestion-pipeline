# FinQA retrieval (1147 test questions)

Run 20261027-0036. Retrieval only, no LLM. Corpus: all 2,789 unique FinQA pages (train + dev + test pooled), so every question searches every page. Units are scored as retrieved, before any passage expansion. Unit sets: paper-style 6a4844d35a90cc37, ours cd86409dbf657f8b. bge-m3 embeddings; full stack = dense + sparse + exact terms, merged by RRF, top 30 reranked.

## Results

| Set-up | Units | any-hit R@5 | R@10 | R@20 | item recall R@5 | R@10 | R@20 | MRR@10 |
|---|---|---|---|---|---|---|---|---|
| dense | paper-style | 42.9 | 50.0 | 58.0 | 35.0 | 41.2 | 49.0 | 32.1 |
| full + MiniLM | paper-style | 48.2 | 57.4 | 67.7 | 40.3 | 48.6 | 58.1 | 34.3 |
| full + bge-reranker-v2-m3 | paper-style | 57.0 | 64.5 | 71.3 | 47.9 | 55.4 | 61.9 | 43.9 |
| dense | ours | 50.5 | 58.2 | 66.0 | 40.9 | 47.8 | 54.9 | 38.3 |
| full + MiniLM | ours | 49.0 | 60.1 | 70.0 | 40.1 | 49.4 | 58.7 | 36.3 |
| full + bge-reranker-v2-m3 | ours | 58.6 | 66.6 | 72.2 | 48.2 | 55.7 | 60.9 | 46.2 |

Item recall R@k: the share of a question's gold items in the top k (the paper's definition). Any-hit R@k: at least one gold item in the top k. MRR@10: 1 / rank of the first gold item.

Gold items held by no unit, counted as misses: paper-style 0; ours 1 (JPM/2018/page_90.pdf-1: text_21: a sentence longer than two chunks). No question was dropped. Test questions were used only for scoring; nothing was tuned on them.

## Read-out

- **Best set-up:** full + bge-reranker-v2-m3 on our units, item recall 48.2 / 55.7 / 60.9: 14 to 23 points below the paper's claimed Naive RAG.
- **Reranker (paper-style units):** bge-reranker-v2-m3 instead of MiniLM moves item recall R@5 from 40.3 to 47.9 and MRR@10 from 34.3 to 43.9.
- **Reranker (our units):** bge-reranker-v2-m3 instead of MiniLM moves item recall R@5 from 40.1 to 48.2 and MRR@10 from 36.3 to 46.2.
- **Unit size:** with dense search alone, our smaller units find more (40.9 vs 35.0 item recall R@5); with the full stack they differ by at most 0.3 points.
- **Where it fails:** with the default set-up the right page is in the top 20 for 86.0% of questions and among the 30 reranked candidates for 89.9%; the gold rows and sentences are then out-ranked by similar rows from other companies and years (many questions do not name the company, and a table row such as "2018 | 301" carries little context on its own).
- **Exact-term matching** barely matters here: FinQA's text has no comma-formatted figures or ids to match as written.

## Next to FinAgent-RAG (Table 11, arXiv 2605.05409v1)

| Comparison point | Paper, item recall R@5 / R@10 / R@20 | Ours, item recall R@5 / R@10 / R@20 |
|---|---|---|
| Naive RAG vs our dense only, paper-style units | 62.3 / 74.8 / 83.5 | 35.0 / 41.2 / 49.0 |
| Naive RAG vs our dense only, our units | 62.3 / 74.8 / 83.5 | 40.9 / 47.8 / 54.9 |
| Iteration 1 vs our best (full + bge-reranker-v2-m3, our units) | 69.7 / 80.6 / 88.1 | 48.2 / 55.7 / 60.9 |

| Paper set-up | R@5 | R@10 | R@20 |
|---|---|---|---|
| Naive RAG (bge-base-en-v1.5 dense, FAISS) | 62.3 | 74.8 | 83.5 |
| Advanced RAG (dense + BM25 + query expansion) | 68.1 | 79.4 | 87.2 |
| FinAgent-RAG iteration 1 (fine-tuned retriever) | 69.7 | 80.6 | 88.1 |
| FinAgent-RAG iteration 3 (LLM query refinement; not comparable) | 82.3 | 90.1 | 94.3 |

Caveats:

- The paper was withdrawn by its authors "due to significant methodological errors in the experimental design"; these are claimed numbers from version 1, and its code is not released.
- Its corpus is "52,847 passages from 8,281 earnings report pages"; FinQA has 8,281 questions but only 2,789 unique pages, so pages may have been indexed once per question (duplicates). Which splits it pooled is not stated. Ours: 2,789 unique pages, all splits.
- Its retriever is bge-base-en-v1.5 with FAISS (Naive RAG) or a fine-tuned retriever (iteration 1); ours is bge-m3, not fine-tuned on FinQA. Iterations 2-3 refine the question with an LLM and are not comparable to a single-pass retriever.
- FinQA's text is lowercased and pre-tokenised ("$ 5735", no thousands separators), unlike the PDFs this pipeline reads.

## How often the right page is retrieved

| Set-up | Units | right page in top 5 (%) | in top 20 (%) |
|---|---|---|---|
| dense | paper-style | 53.9 | 69.9 |
| full + MiniLM | paper-style | 62.8 | 81.6 |
| full + bge-reranker-v2-m3 | paper-style | 69.8 | 83.3 |
| dense | ours | 64.7 | 80.7 |
| full + MiniLM | ours | 66.3 | 86.0 |
| full + bge-reranker-v2-m3 | ours | 70.5 | 85.7 |

The right page is among the 30 candidates handed to the reranker for 85.7% (paper-style units), 89.9% (our units) of questions.

## Five misses of the default set-up (full + MiniLM, our units): what came back instead

**ETR/2016/page_23.pdf-2**: what is the net change in net revenue during 2015 for entergy corporation?

- Gold: `2014 net revenue | $ 5735` / `2015 net revenue | $ 5829`
- Retrieved #1: (ETR/2015/page_18.pdf) as shown in the table above , net revenue for entergy wholesale commodities decreased by approximately $ 558 million in 2015 primarily due to : 2022 lower realized wholesale energy prices , primarily 
- Retrieved #2: (ETR/2016/page_382.pdf) entergy mississippi requested a net increase in revenue of $ 49 million for bills rendered during calendar year 2015 , including $ 30 million resulting from new depreciation rates to update the estima
- Retrieved #3: (ETR/2016/page_24.pdf) as shown in the table above , net revenue for entergy wholesale commodities decreased by approximately $ 558 million in 2016 primarily due to : 2022 lower realized wholesale energy prices , primarily 

**FIS/2010/page_70.pdf-2**: what portion of total purchase price is related to stock awards?

- Gold: `value of metavante stock awards | 121.4` / `total purchase price | $ 4187.8`
- Retrieved #1: (SNPS/2012/page_64.pdf) the company allocated the total purchase consideration of $ 316.6 million ( including $ 4.6 million related to stock awards assumed ) to the assets acquired and liabilities assumed based on their resp
- Retrieved #2: (SNPS/2009/page_59.pdf) the fair value of stock options assumed was determined using a black-scholes valuation model . the fair value of stock awards vested or earned of $ 4.2 million was included as part of the purchase pri
- Retrieved #3: (UPS/2014/page_61.pdf) cash outflows related to the repurchase of shares to satisfy tax withholding obligations on vested employee stock awards were $ 224 , $ 253 and $ 234 million for 2014 , 2013 and 2012 , respectively . 

**MAS/2017/page_27.pdf-2**: what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2017?

- Gold: `masco | $ 138.48 | $ 155.26 | $ 200.79 | $ 227.08 | $ 318.46` / `s&p 500 index | $ 132.04 | $ 149.89 | $ 151.94 | $ 169.82 | $ 206.49`
- Retrieved #1: (BLL/2012/page_31.pdf) shareholder return performance the line graph below compares the annual percentage change in ball corporation fffds cumulative total shareholder return on its common stock with the cumulative total re
- Retrieved #2: (BLL/2011/page_29.pdf) shareholder return performance the line graph below compares the annual percentage change in ball corporation fffds cumulative total shareholder return on its common stock with the cumulative total re
- Retrieved #3: (BLL/2010/page_28.pdf) page 15 of 100 shareholder return performance the line graph below compares the annual percentage change in ball corporation 2019s cumulative total shareholder return on its common stock with the cumu

**GS/2015/page_188.pdf-2**: what percentage of future minimum rental payments are due in 2018?

- Gold: `2018 | 301` / `total | $ 2575`
- Retrieved #1: (AWK/2018/page_178.pdf) as of december 31 , 2018 , the minimum annual future rental commitment under the operating leases for the portion of the facilities funded by the partners that have initial or remaining non-cancelable
- Retrieved #2: (ANET/2017/page_156.pdf) as of december 31 , 2017 , the aggregate future minimum payments under non-cancelable operating leases consist of the following ( in thousands ) : years ending december 31 . rent expense for all opera
- Retrieved #3: (ADI/2019/page_85.pdf) total rental expense under operating leases was approximatelya $ 92.3 million in fiscal 2019 , $ 84.9 million in fiscal 2018 and $ 58.8 million in fiscal 2017 . the following is a schedule of futureff

**CMCSA/2015/page_112.pdf-2**: what was the change in unrecognized tax benefits from the end of 2014 to the end of 2015?

- Gold: `balance december 31 | $ 1136 | $ 1171 | $ 1701`
- Retrieved #1: (AMT/2016/page_125.pdf) during the years ended december 31 , 2016 , 2015 and 2014 , the statute of limitations on certain unrecognized tax benefits lapsed and certain positions were effectively settled , which resulted in a 
- Retrieved #2: (RSG/2015/page_126.pdf) republic services , inc . notes to consolidated financial statements 2014 ( continued ) the following table summarizes the activity in our gross unrecognized tax benefits for the years ended december 
- Retrieved #3: (KIM/2014/page_131.pdf) open years range from 2008 through 2014 and vary by jurisdiction and issue . the aggregate changes in the balance of unrecognized tax benefits for the years ended december 31 , 2014 and 2013 were as f
