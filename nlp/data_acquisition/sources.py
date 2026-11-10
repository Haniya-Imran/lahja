"""Public datasets used for training, pinned to exact Git commits so downloads are reproducible.

Every file is fetched from the original public GitHub repository at a fixed commit and
checked against a SHA-256 hash recorded on first download (data/raw/manifest.json).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    language: str              # dominant language/script of the source
    url: str                   # raw file URL at a pinned commit
    archive_member: str | None  # file inside a .tar.gz, if the URL is an archive
    homepage: str
    license: str
    citation: str
    labels: str
    notes: list[str] = field(default_factory=list)


GH = "https://raw.githubusercontent.com"

SOURCES: list[Source] = [
    Source(
        key="roman_urdu_uci",
        name="Roman Urdu Data Set (Sharf, UCI)",
        language="roman_urdu",
        url=f"{GH}/Smat26/Roman-Urdu-Dataset/762ba24484024855798ef15d248e45f12a410de3/Dataset/Roman%20Urdu%20DataSet.csv",
        archive_member=None,
        homepage="https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set",
        license="CC BY 4.0 (UCI listing); GitHub mirror distributed under GPL-3.0",
        citation="Sharf, Z. (2018). Roman Urdu Data Set. UCI Machine Learning Repository. https://doi.org/10.24432/C58325",
        labels="Positive / Negative / Neutral (manually tagged)",
        notes=["~20k sentences from e-commerce reviews, Facebook and Twitter comments.",
               "Known to contain some noisy or inconsistent labels; one 'Neative' typo."],
    ),
    Source(
        key="daraz_reviews",
        name="Daraz Roman Urdu product reviews",
        language="roman_urdu_mixed",
        url=f"{GH}/mirfan899/Urdu/818f2b5f9e584b10aaca25621fa7b837a18ecd2a/sentiment/daraz_products_reviews.csv.tar.gz",
        archive_member="daraz_products_reviews.csv",
        homepage="https://www.kaggle.com/datasets/naveedhn/daraz-roman-urdu-reviews",
        license="Mirror repository under MIT; original Kaggle upload by naveedhn (license not stated on the mirror)",
        citation="naveedhn. Daraz Roman Urdu Reviews. Kaggle. Mirrored in github.com/mirfan899/Urdu.",
        labels="Positive / Negative / Neutral plus a spam flag",
        notes=["Real Pakistani e-commerce reviews; Roman Urdu, English and mixtures.",
               "Rows flagged as spam are removed."],
    ),
    Source(
        key="urdu_products",
        name="Urdu products & services sentiment",
        language="urdu",
        url=f"{GH}/mirfan899/Urdu/818f2b5f9e584b10aaca25621fa7b837a18ecd2a/sentiment/products_sentiment_urdu.csv.tar.gz",
        archive_member="sentiment/products_sentiment_urdu.csv",
        homepage="https://github.com/mirfan899/Urdu",
        license="MIT (repository license)",
        citation="Irfan, M. (2019). Urdu datasets. github.com/mirfan899/Urdu",
        labels="pos / neg / neu",
        notes=["Urdu-script sentences about products and services from social media and the web."],
    ),
    Source(
        key="urdu_sentiment_corpus",
        name="Urdu Sentiment Corpus v1.0",
        language="urdu",
        url=f"{GH}/MuhammadYaseenKhan/Urdu-Sentiment-Corpus/8718cf9d03f16c90006f4e61fb5009164f5cf25f/urdu-sentiment-corpus-v1.tsv",
        archive_member=None,
        homepage="https://github.com/MuhammadYaseenKhan/Urdu-Sentiment-Corpus",
        license="Not stated by the authors (research corpus; cite the paper)",
        citation="Khan, M. Y. & Nizami, M. S. (2020). Urdu Sentiment Corpus (v1.0). IEEE ICISCT 2020.",
        labels="P / N / O (O = objective, mapped to neutral)",
        notes=["1,000 Urdu tweets, mostly political; only 20 objective tweets."],
    ),
    Source(
        key="english_finegrained",
        name="Fine-grained sentence sentiment (Täckström & McDonald)",
        language="english",
        url=f"{GH}/oscartackstrom/sentence-sentiment-data/da58c4d997c31aec2e61dc0178eb0863330bb21e/data/finegrained.txt",
        archive_member=None,
        homepage="https://github.com/oscartackstrom/sentence-sentiment-data",
        license="Free for any friendly purpose; cite the ECIR 2011 paper",
        citation="Täckström, O. & McDonald, R. (2011). Discovering fine-grained sentiment with latent variable structured prediction models. ECIR 2011.",
        labels="pos / neg / neu (mix and nr dropped)",
        notes=["Sentences from 294 product reviews (books, DVDs, electronics, music, video games)."],
    ),
] + [
    Source(
        key=f"english_uci_{name}",
        name=f"Sentiment Labelled Sentences — {name.title()} (Kotzias et al., UCI)",
        language="english",
        url=f"{GH}/atharvajk98/UCI-Sentiment-Analysis/122d6362f5c126836c0e620f06ebbb4195ec812a/Dataset/{fname}",
        archive_member=None,
        homepage="https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
        license="CC BY 4.0",
        citation="Kotzias, D., Denil, M., de Freitas, N. & Smyth, P. (2015). From group to individual labels using deep features. KDD 2015.",
        labels="1 positive / 0 negative (binary)",
        notes=["1,000 review sentences; binary only, so it adds no neutral examples."],
    )
    for name, fname in (("amazon", "amazon_cells_labelled.txt"), ("yelp", "yelp_labelled.txt"), ("imdb", "imdb_labelled.txt"))
]

# Roman Urdu dictionary used only to help the language detector recognise Roman Urdu words.
ROMAN_URDU_DICTIONARY = (
    f"{GH}/Smat26/Roman-Urdu-Dataset/762ba24484024855798ef15d248e45f12a410de3/Dictionary/Roman-Urdu-Dictionary.txt"
)
