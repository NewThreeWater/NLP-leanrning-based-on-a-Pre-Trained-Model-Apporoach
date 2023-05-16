import nltk
nltk.download() # 下载NLTK提供的语料库和词典资源

from nltk.corpus import stopwords
stopwords.words('english')
print(stopwords.words('chinese')) # 引用nltk停用词,具体模块介绍：https://www.nltk.org/

# NLTK提供了多种语料库（文本数据集），如图书、电影评论和聊天记录等，它们可以
# 被分为两类，即未标注语料库（又称生语料库或生文本，Raw text）和人工标注语料库
# （Annotated corpus）。
from nltk.corpus import gutenberg
gutenberg.raw('austen-emma.txt')
print(gutenberg.raw('austen-emma.txt')) # 调用NLTK提供的相应功能以获得古腾堡（Gutenberg）语料库

from nltk.corpus import sentence_polarity
[(sentence,category)
 for category in sentence_polarity.categories()
     for sentence in sentence_polarity.sents(categories=category)]
# 在句子极性语料库（sentence_polarity）中，包含了10，662条来自电影领域的用户评论句子
# 以及相应的极性信息（褒义或贬义）。

# WordNet 是普林斯顿大学构建的英文语义词典（也称作辞典，
# Thesaurus），其主要特色是定义了同义词集合（Synset），每个同义词集合由具有相同
# 意义的词义组成。此外，WordNet为每一个同义词集合提供了简短的释义（Gloss），同
# 时，不同同义词集合之间还具有一定的语义关系。
from nltk.corpus import wordnet
syns = wordnet.synsets('bank') # 返回bank的全部词义的同义词
syns[0].name() # 返回bank第一个词义的名称
syns[0].definition() # 返回bank第一个词义的定义
syns[0].examples() # 返回bank第一个词义的示例
syns[0].hypernyms() # 返回bank第一个词义的上位同义词集合
dog = wordnet.synset('dog.n.01')
cat = wordnet.synset('cat.n.01')
dog.wup_similarity(cat) # 计算两个同义词集合之间的wu-palmer相似度

from nltk.corpus import sentiwordnet
sentiwordnet.senti_synset('good.a.01')
# 。SentiWordNet（Sentiment WordNet）是基于WordNet标注的词
# 语（更准确地说是同义词集合）情感倾向性词典，它为WordNet中每个同义词集合人工标
# 注了三个情感值，依次是褒义、贬义和中性。

# 利用nltk分句
# 通常一个句子能够表达完整的语义信息，因此在进行更深入的自然语言处理之前，往
# 往需要将较长的文档切分成若干句子，这一过程被称为分句。一般来讲，一个句子结尾具
# 有明显的标志，如句号、问号和感叹号等，因此可以使用简单的规则进行分句。然而，往
# 往存在大量的例外情况，如在英文中，句号除了可以作为句尾标志，还可以作为单词的一
# 部分（如“Mr.”）。NLTK提供的分句功能可以较好地解决此问题。下
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
text = gutenberg.raw('austen-emma.txt')
sentences = sent_tokenize(text) # 对小说进行分词
print(sentences[100]) # 显示效果如何

# 一个句子是由若干标记（Token）按顺序构成的，其中标记既可以是一个词，也可以
# 是标点符号等，这些标记是自然语言处理最基本的输入单元。将句子分割为标记的过程叫
# 作标记解析（Tokenization）。
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
text = gutenberg.raw('austen-emma.txt')
sentences = sent_tokenize(text) # 对小说进行分词
print(sentences[100]) # 显示效果如何
from nltk.tokenize import word_tokenize
print(word_tokenize(sentences[100]))

# 词性是词语所承担的语法功能类别，如名词、动词和形容词等，因此词性也被称为词
# 类。很多词语往往具有多种词性，如“fire”，即可以作名词（“火”），也可以作动词
# （“开火”）。词性标注就是根据词语所处的上下文，确定其具体的词性。
from nltk import pos_tag
from nltk.tokenize import word_tokenize
print(pos_tag(word_tokenize('i am your father')))
# 中文不太能分辨 名词（NN）动词（VBP） 词性标记采用宾州树库（Penn Treebank）的标注标准
# 除了以上介绍的分句、标记解析和词性标注，NLTK还提供了其他丰富的自然语言处理
# 工具，包括命名实体识别、组块分析（Chunking）和句法分析等。






