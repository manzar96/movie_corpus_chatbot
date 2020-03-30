# slp

[![Build Status](https://travis-ci.org/georgepar/slp.svg?branch=master)](https://travis-ci.org/georgepar/slp)
[![Maintainability](https://api.codeclimate.com/v1/badges/d3ad9729ad30aa158737/maintainability)](https://codeclimate.com/github/georgepar/slp/maintainability)

Utils and modules for NLP, audio and multimodal processing using sklearn and pytorch


Training:
1. Make a venv and use pip install -r requirements.txt to install required
 packages.
1. Add Movie Triples data to: ./data/MovieTriples
2. Add emebddings file to: ./cache
3. Run in terminal: export PYTHONPATH=./ (in cloned root dir)
4. python experiments/hred/training/hred_movie_triples.py -epochs 80 -lr 0
.0005 -ckpt ./checkpoints/hred -emb_dim 300

You can also add more options:

-shared (to use shared weights between encoder and decoder)

-shared_emb (to use shared embedding layer for encoder and decoder)

-emb drop 0.2 (embeddings dropout)

-encembtrain (to train encoder embeddings)

-decembtrain (to train decoder embeddings)

(see argparser in experiments/hred/training/hred_movie_triples.py for more)

Running experiments:

1. python experiments/hred/training/hred_movie_triples.py -epochs 80 -lr 0
.0005 -ckpt ./checkpoints/<vale oti thes edw> -emb_dim 300 -decr_tc_ratio
 -encembtrain -decembtrain -shared_emb

2.  python experiments/hred/training/hred_movie_triples.py -epochs 80 -lr 0
.0005 -ckpt ./checkpoints/<vale oti thes edw> -emb_dim 300 