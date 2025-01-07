import os
import sys


def include():
    sys.path.append(os.path.abspath(os.path.join('..', '2ndRealDataJournalWithoutGit/bestResults')))
    sys.path.append(os.path.abspath(os.path.join('..', '2ndRealDataJournalWithoutGit/checkpoint')))
    sys.path.append(os.path.abspath(os.path.join('..', '2ndRealDataJournalWithoutGit/data')))
    # sys.path.append(os.path.abspath(os.path.join('ai_core/object_detection','plate_character')))
#    sys.path.append(os.path.abspath(os.path.join('..', 'common')))
