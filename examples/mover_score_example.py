from moverscore_v2 import get_idf_dict,word_mover_score
from collections import defaultdict
from moverscore_v2 import plot_example


translations = ["Hi! What's up?"]
references = ["hi how are you?"]

# idf_dict_hyp = get_idf_dict(translations)
# idf_dict_ref = get_idf_dict(references)
idf_dict_hyp = defaultdict(lambda: 1.)
idf_dict_ref = defaultdict(lambda: 1.)

scores1 = word_mover_score(references, translations, idf_dict_ref,
                          idf_dict_hyp, stop_words=[], n_gram=1,
                          remove_subwords=True)


translations = ["Hi! What's up?","I am good"]
references = ["hi how are you?","I am well"]

scores3 = word_mover_score(references, translations, idf_dict_ref,
                          idf_dict_hyp, stop_words=[], n_gram=1,
                          remove_subwords=True)


# reference = 'they are now equipped with air conditioning and new toilets.'
# translation = 'they have air conditioning and new toilets.'
# plot_example(True, reference, translation)
#
# import ipdb;ipdb.set_trace()
