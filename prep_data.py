# Created by Bhaskar at 1/5/19
from os import listdir
import re
pattern = re.compile(r"(?u)[a-zA-Z0-9]+")


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename)
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights


# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story': story, 'highlights': highlights})
    return stories


# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    # table = string.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index + len('(CNN)'):]
        # tokenize on white space
        line = pattern.findall(line)
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        # line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


# load stories
directory = 'dailymail/stories/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))

# clean stories
f1 = open("stories1.txt", 'w')
f2 = open("summary1.txt", 'w')
for example in stories:
    example['story'] = clean_lines(example['story'].split('\n'))
    example['highlights'] = clean_lines(example['highlights'])
    f1.write(" ".join(example['story']))
    f1.write("\n")
    f2.write(" ".join(example['highlights']))
    f2.write("\n")
f1.close()
f2.close()

story = open("stories1.txt").readlines()
summ = open("summary1.txt").readlines()
train_story = story[0:210000]
train_summ = summ[0:210000]

eval_story = story[210000:]
eval_summ = summ[210000:]

# test_story = story[91579:92579]
# test_summ = summ[91579:92579]

with open("data1/train_story.txt", 'w') as f:
    f.write("\n".join(train_story))

with open("data1/train_summ.txt", 'w') as f:
    f.write("\n".join(train_summ))

with open("data1/eval_story.txt", 'w') as f:
    f.write("\n".join(eval_story))

with open("data1/eval_summ.txt", 'w') as f:
    f.write("\n".join(eval_summ))
#
# with open("test_story.txt", 'w') as f:
#     f.write("\n".join(test_story))
#
# with open("test_summ.txt", 'w') as f:
#     f.write("\n".join(test_summ))
#     """