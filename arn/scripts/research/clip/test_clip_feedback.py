"""This is a test of CLIPFeedbackInterpreter."""

from arn.model.feedback import CLIPFeedbackInterpreter
from arn.data.kinetics_unified import KineticsUnified

# TODO We do not have ANY PAR data that has the both the PAR labels AND PAR
# feedback. So we need to mock it with our Kinetics data.

# TODO Load sam's CSV of 1K samples of Kinetics700_2020 with PAR labels and
# that includes the top 5 predictions of X3D, or some other model that is a
# stand in for ID3 trained on Kinetics 600.

# Future NOTE, load in top 5 predictions of CLIP as well because that is more
# ideal feedback that is probably more correct and i think is also roughly the
# same to ID3 baseline.

interpreter = CLIPFeedbackInterpreter(**args.feedback_interpreter)
mock_data = KineticsUnified(**args.kinetics_unified)

# TODO assess the novelty recognition of the interpretter alone

# TODO assess the novelty recognition of the predictor w/o interpretter

# TODO assess the novelty recognition of the predictor w/ interpretter

# TODO assess the novelty recognition of the full predictor overtime.
