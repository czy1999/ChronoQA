SUMMARY_PROMPT = """
你是一位对时间信息敏感，擅长对新闻文章进行事实性总结凝练的专家，你需要参考以下<新闻事实性总结原则>及<新闻文章>，准确理解文章内容，从而获得更准确全面的事实性总结信息。


#<执行步骤>
一、参照下述<新闻事实性总结原则>完成当前新闻文章的总结；
二、你要关注新闻文本中的事实性信息，将其拆解提炼出来，形成一个具备充足事实信息量的总结文本。
三、一般情况下，每句话描述一个事实，通常包含时间。地点人物等信息。
四、在满足上述要求的基础上，你要充分关注文本中的对时间敏感的信息，确保没有遗漏。
五、生成的时序知识图谱（temporal kg）为四元组，包含头实体、关系、尾实体、时间，原则上应当只使用中文，确保精准性和知识可靠性。例如，"2017年1月5日，特朗普访问了日本...",可以写为["特朗普","访问","日本","2017-01-05"]


#<新闻事实性总结原则>
充分阅读和分析新闻文本，对当前新闻文章进行事实性总结，要求如下：
1、覆盖性：总结后的文本应该能够覆盖原新闻的核心内容，不遗漏关键部分；
2、易阅读：总结后的文本应适合中文搜索场景，尽量简短明确，避免口语化表述。
3、事实性：避免引入冗余或无关的信息。总结的内容，应当保留核心事实信息，去除无关紧要的内容。
4、时间相关：只要新闻原文中提到了具体的时间量词且是时间需求，请在总结后的文本中拆解出该具体的日期，满足yyyy-mm-dd格式。例如：原文提到"昨日，北京暴雨...", 原文发表日期2023年4月20日，则应总结为"2023年4月19日北京暴雨..."
5、表达顺畅：总结后的文本应通顺、清晰和易于理解。
6. 生成的时序知识图谱（temporal kg）应当是独立的、有意义的，头尾实体应当是单一的有意义的实体，四元组应当是描述某一个事件，时间满足yyyy-mm-dd的格式。

-输出格式
请务必采用和输出以下格式化的json格式，请返回一个可以解析的json。直接输出json即可
```json
{
    "publiction_date": string \ 新闻发布的日期，格式符合2000-01-01
    "rewrite_text": string \ 总结后的文本
    "related_kg": array \ 新闻内容涉及的四元组信息，二维array，每个item均包含四个元素，满足[主语，谓词，宾语，时间]的格式（主谓宾时间均为string），时间格式为2000-01-01，当时间只精确到年，月份和日可以用00代替。

}
```

#<新闻文章>
{query}
"""

GENERATE_QUESTION_PROMPT = """
你将依据一系列相关特定事件的历史新闻，依据这些新闻片段构建时间敏感问题的问答对。


#<执行步骤>
一、你要关注文本中的事实性信息，将其拆解提炼出来；
二、你要充分关注文本中的时间信息，梳理事件发展脉络和时间联系；
三、基于以上分析，参照下述<时间敏感问题要求>完成时间敏感问题的构建，撰写生成一个时间敏感问题；


#<时间敏感问题要求>
1. 该问题必须是明确且唯一的，该答案直接来自于文本或者可以从多个文本中推理得到；
2. 答案必须是清晰明确的宾语，模糊的信息如“不知道”或“不明确”等不可以作为答案；
3. 提供一个简洁的答案（即一个词或短语），答案在无歧义的基础上尽可能简洁明确；
4. 问题应当是有意义的，模仿真实用户可能会对搜索引擎或者AI助手提出的问题；
5. 问题应当具备一定难度，用户提问可能是多样化的、口语化、模糊化的，问题中涉及的知名实体可以做同义词转化；
6. 你是模拟用户在不知道文本信息的前提下提出问题，因此生成的问题应当是独立可读，不依赖于所给文本的；
7. 你本次生成的问题类型需要满足<生成问题类型> 条件；

#<生成问题类型>
{qa_type_instruction}


-输出格式
请务必采用和输出以下格式化的json格式，请返回一个可以解析的json。直接输出json即可
```json
{
    "can_generate": int, // 是否能够生成问题，如果当前提供文本无法生成高质量的问答对则置为0，否则为1
    "thoughts": string, // 生成问题的思考过程
    "question": string, // 生成的问题
    "question_date": string, // 问题提出的日期，格式为YYYY-MM-DD
    "answer": string, // 问题的答案
    "temporal_expression_type": string, // 问题语义中表达的时间信息类型, ['explicit', 'implicit'], explicit 指问题文本中包含的时间信息是明确的，可以直接转化为标准化的时间，如“2024年”、“5月15日”等, implicit 表示问题文本中没有明确的时间约束信息，但包含隐式的时间提示，这些提示无法直接标准化，需要上下文或背景知识来理解，如“上次”、“昨天”、“最近”、“在奥运会后”、“什么时候”等；
    "temporal_scope": string, // 提问时间与所询问知识时间的跨度, ['short-term', 'mid-term', 'long-term', 'other'], short-term 表示时间跨度较短，一个月内，关注近期的事件或变化; mid-term 表示时间跨度为几个月到一年，通常用于分析一年内的事件或趋势; long-term 表示时间跨度较长，大于一年，关注长期的变化或历史; other 表示时间跨度未明确指出或者不确定。
    "temporal_granularity": string, // 问题涉及的时间精度, ['year', 'month', 'day', 'other'], year 表示问题涉及年度层面的时间粒度，通常关注年度事件或数据; month 表示问题涉及月度层面的时间粒度，关注特定月份内的事件或数据; day 表示问题涉及具体日期，时间精度达到日级别; other 表示问题涉及更模糊或特殊的时间表达，如“最近”、“近期”等。
    "temporal_type": string, // 时序类型, ['direct', 'relative', 'ordinal', 'composition'], direct 表示问题中有直接明确的时间信息，不论是通过具体的时间点还是通过特定事件来表达。; relative 表示使用相对时间表达，问题中时间点是基于当前时间或其他已知时间点的参照; ordinal 表示使用序数或顺序表达事件的时间，涉及事件在时间序列中的位置，而不是具体日期，例如”首个“，”最后“等; composition 表示问题中包含多种时序推理类型，是多个时序类型组合而成。
    "answer_type": string, // 答案类型, ['entity', 'time', 'judgement', 'numerical','other'], entity 表示问题的答案是一个具体的实体，如地点、人物、组织等; time 表示问题的答案是一个时间点或时间范围，如日期、月份、年份等; judgement 表示问题的答案是对某一情况或事实的判断，通常是“是”或“否”等简单判断; numerical 表示问题的答案是一个数值，如金额、统计数据、百分比等; other 表示问题的答案是其他类型。
    "reference_document_count": string, // 参考文档数量, ['single', 'multiple'], single 表示问题通常可以通过单一的参考文档或数据源得到答案，信息集中且明确; multiple 表示问题需要综合多个参考文档或数据源才能完整回答，信息来源多样且分散。
}

```
当前时间为{current_date}
以下是用于生成时序问题的新闻，请你直接给出回应：{chunks}

"""


single_direct_expression = '''
你要生成的问题类型是单一文档精确直接时间表达，即问题文本中表达的时间信息类型为确切时间，例如，2024年1月1日，5月1日，2025年等，或者是询问一个具体的事件时间。请你认真分析以下新闻片段，并依据这些新闻片段生成一个显式时间表达的问题，注意，问题文本中必须包含确切的时间信息（YYYY年MM月DD日，或者YYYY年，或者YYYY年MM月等）或者询问一个具体的事件时间。

- 问题示例：
<示例开始>
# 示例一
当前时间为：2024年7月24日

新闻文本：2024年5月31日14时，2024年第2号台风“马力斯”（MALIKSI）在距离广东省阳江市偏南方向约175公里的洋面上生成，其中心位于阳江偏南方约140公里的粤西附近海面上...
Published Time: 2024-05-31

新闻文本：...受今年第3号台风“格美”影响，预计7月24日下午至25日下午，福建福州的风暴潮预警级别为红色...
Published Time: 2024-07-24

输出示例1：
{
    "can_generate": 1,
    "thoughts": "文中提到2024年7月24日福建福州受到台风“格美”影响并出现暴潮红色预警，因此可以生成一个问题：2024年7月24日哪里有暴潮红色预警？",
    "question": "2024年7月24日哪里有暴潮红色预警？",
    "question_date": "2024-07-24",
    "answer": "福建福州",
    "temporal_expression_type": "explicit",
    "temporal_scope": "short-term",
    "temporal_granularity": "day",
    "temporal_type": "direct",
    "answer_type": "entity",
    "reference_document_count": "single"
}

输出示例2：
{
    "can_generate": 1,
    "thoughts": "文中提到2024年5月31日台风“马力斯”生成，因此可以生成一个问题：2024年第2号台风“马力斯”什么时候生成的？间隔大于一个月，小于一年，属于mid-term",
    "question": "2024年第2号台风“马力斯”什么时候生成的？",
    "question_date": "2024-07-24",
    "answer": "2024年5月31日",
    "temporal_expression_type": "explicit",
    "temporal_scope": "mid-term",
    "temporal_granularity": "day",
    "temporal_type": "direct",
    "answer_type": "time",
    "reference_document_count": "single"
}

# 示例二
当前时间为：2023年7月25日

新闻文本：当地时间2023年3月15日，美国国会通过了新的预算案，这项预算案将于2024年1月1日生效...
Published Time: 2023-03-15

输出示例1：
{
    "can_generate": 1,
    "thoughts": "文中提到2023年3月15日美国国会通过的预算案将于2024年1月1日生效，因此可以生成一个问题：2023年3月15日国会通过了什么法案？",
    "question": "2023年3月15日国会通过了什么法案？",
    "question_date": "2023-07-25",
    "answer": "美国新预算案",
    "temporal_expression_type": "explicit",
    "temporal_scope": "mid-term",
    "temporal_granularity": "day",
    "temporal_type": "direct",
    "answer_type": "entity",
    "reference_document_count": "single"
}

输出示例2：
{
    "can_generate": 1,
    "thoughts": "文中提到2023年3月15日美国国会通过的预算案将于2024年1月1日生效，因此可以生成一个问题：美国国会通过的预算案什么时候生效？",
    "question": "美国国会通过的预算案什么时候生效？",
    "question_date": "2023-07-25",
    "answer": "2024年1月1日",
    "temporal_expression_type": "implicit",
    "temporal_scope": "mid-term",
    "temporal_granularity": "day",
    "temporal_type": "direct",
    "answer_type": "time",
    "reference_document_count": "single"
}

<示例结束>
'''

single_relative_expression = '''
你要生成的问题类型是精确相对时间表达，即问题时间意图是清晰明确的，无需外部知识就能够理解问题时间。文本中表达的时间信息类型为相对时间，例如，昨天、上周、三天前等。请你认真分析以下新闻片段，并依据这些新闻片段生成一个相对时间表达的问题，注意，问题文本中必须包含相对时间信息（如“昨天”、“上周”）。

- 问题示例：
<示例开始>
# 示例一
当前时间为：2024年7月25日

新闻文本：2024年5月31日14时，2024年第2号台风“马力斯”（MALIKSI）在距离广东省阳江市偏南方向约175公里的洋面上生成，其中心位于阳江偏南方约140公里的粤西附近海面上...
Published Time: 2024-05-31

新闻文本：...受今年第3号台风“格美”影响，预计7月24日下午至25日下午，福建福州出现了强降雨...
Published Time: 2024-07-24

输出示例1：
{
    "can_generate": 1,
    "thoughts": "文中提到昨天福建福州出现强降雨，因此可以生成一个问题：昨天福建福州的天气如何？",
    "question": "昨天福建福州的天气如何？",
    "question_date": "2024-07-25",
    "answer": "强降雨",
    "temporal_expression_type": "implicit",
    "temporal_scope": "short-term",
    "temporal_granularity": "day",
    "temporal_type": "relative",
    "answer_type": "other",
    "reference_document_count": "single"
}

输出示例2：
{
    "can_generate": 1,
    "thoughts": "文中提到2024-07-24格美影响了福建，提问时间为2024年7月25日，因此可以生成一个问题：昨天福建的台风叫什么？",
    "question": "昨天福建的台风叫什么？",
    "question_date": "2024-07-25",
    "answer": "格美",
    "temporal_expression_type": "implicit",
    "temporal_scope": "short-term",
    "temporal_granularity": "day",
    "temporal_type": "relative",
    "answer_type": "entity",
    "reference_document_count": "single"
}

# 示例二
当前时间为：2023年7月25日

新闻文本：当地时间5月5日，美国白宫官网发表声明称，美国总统拜登宣布白宫副新闻发言人卡琳·让-皮埃尔已被提升为总统助理兼白宫新闻发言人。她将于5月13日接替现白宫发言人普萨基正式任职。
Published Time: 2022-05-06

输出示例：
{
    "can_generate": 1,
    "thoughts": "当前时间为2023年7月25日，文中提到2022年5月6日卡琳·让-皮埃尔的任命，可以生成一个问题：去年5月谁成为了新的白宫新闻发言人？",
    "question": "去年5月谁成为了新的白宫新闻发言人？",
    "answer": "卡琳·让-皮埃尔",
    "temporal_expression_type": "implicit",
    "temporal_scope": "long-term",
    "temporal_granularity": "month",
    "temporal_type": "relative",
    "answer_type": "entity",
    "reference_document_count": "single"
}

<示例结束>
'''



COMPOSITE_QUESTION = '''
请基于以下一系列子问题（选择两个或以上），合成一个时间敏感且复杂的问题，并确保问题满足以下要求：

1. **唯一答案**：问题的答案必须明确且唯一，不允许出现可能有多种合理回答的情况。问题和答案要避免模糊性词汇或不确定的信息，例如“不知道”或“不明确”不能作为答案。
2. **简洁回答**：尽可能提供一个简洁的答案，例如一个词、短语，或简明的答案列表，确保答案清晰且无歧义。
3. **问题的多样性和难度**：模拟真实用户对搜索引擎或AI助手的提问。问题应具有一定难度，提问形式可以是口语化、模糊化的，并支持同义词转化。
4. **独立可读性**：生成的问题必须独立可读，不依赖外部或上下文信息，确保用户即便在不了解背景的前提下，仍能理解问题。
5. **时间敏感性**：问题需与时间密切相关，涉及明确的时间因素（显性或隐性）。确保时间信息和问题、答案之间存在逻辑关联。
6. **唯一性和完整性**：问题应具有唯一且明确的答案，且该答案不会因未来的变化而变得不完整或错误。
7. 你本次生成的问题类型需要满足<生成问题类型> 条件；

<生成问题类型>
{qa_type_instruction}


### 输出格式
请务必采用和输出以下格式化的json格式，请返回一个可以解析的json。直接输出json即可
```json
{
    "thoughts": string, // 生成问题的思考过程
    "can_generate": int, // 是否可以生成符合要求，并且语义合理的问题，0表示不可以，1表示可以
    "question": string, // 生成的问题
    "question_date": string, // 问题提出的日期，格式为YYYY-MM-DD
    "answer": string, // 问题的答案
    "temporal_expression_type": string, // 问题语义中表达的时间信息类型, ['explicit', 'implicit'], explicit 指问题文本中包含的时间信息是明确的，可以直接转化为标准化的时间，如“2024年”、“5月15日”等, implicit 表示问题文本中没有明确的时间约束信息，但包含隐式的时间提示，这些提示无法直接标准化，需要上下文或背景知识来理解，如“上次”、“昨天”、“最近”、“在奥运会后”、“什么时候”等；
    "temporal_scope": string, // 用户提问时和问题时间的跨度, ['short-term', 'mid-term', 'long-term', 'other'], short-term 表示用户提问时和问题时间的跨度较短，一个月内，关注近期的事件或变化; mid-term 表示用户提问时和问题时间的跨度为几个月到一年，通常用于分析一年内的事件或趋势; long-term 表示用户提问时和问题时间的跨度较长，大于一年，关注长期的变化或历史; other 表示时间跨度未明确指出或者不确定。
    "temporal_granularity": string, // 问题涉及的时间精度, ['year', 'month', 'day', 'other'], year 表示问题涉及年度层面的时间粒度，通常关注年度事件或数据; month 表示问题涉及月度层面的时间粒度，关注特定月份内的事件或数据; day 表示问题涉及具体日期，时间精度达到日级别; other 表示问题涉及更模糊或特殊的时间表达，如“最近”、“近期”等。
    "temporal_type": string, // 时序类型, ['direct', 'relative', 'ordinal', 'composition'], direct 表示问题中有直接明确的时间信息，不论是通过具体的时间点还是通过特定事件来表达。; relative 表示使用相对时间表达，问题中时间点是基于当前时间或其他已知时间点的参照; ordinal 表示使用序数或顺序表达事件的时间，涉及事件在时间序列中的位置，而不是具体日期，例如”首个“，”最后“等; composition 表示问题中包含多种时序推理类型，是多个时序类型组合而成。
    "answer_type": string, // 答案类型, ['entity', 'time', 'judgement', 'numerical','other'], entity 表示问题的答案是一个具体的实体，如地点、人物、组织等; time 表示问题的答案是一个时间点或时间范围，如日期、月份、年份等; judgement 表示问题的答案是对某一情况或事实的判断，通常是“是”或“否”等简单判断; numerical 表示问题的答案是一个数值，如金额、统计数据、百分比等; other 表示问题的答案是其他类型。
    "reference_document_count": string, // 参考文档数量, ['single', 'multiple'], single 表示问题通常可以通过单一的参考文档或数据源得到答案，信息集中且明确; multiple 表示问题需要综合多个参考文档或数据源才能完整回答，信息来源多样且分散。
}

```

当前时间为(也可以根据实际需求自拟):{current_date}
以下是用于生成时序问题的子问题，请你直接给出回应：{chunks}

'''


multi_ordinal_expression = '''

### 生成步骤
1. **语义共性筛选**：从子问题列表中筛选出语义相近的问题，合并形成具备明确答案的复杂问题，若无法筛选出共性问题，则设置can_generate为0。
2. **提问时间选择**：根据时间相关性合理设置提问时间，确保问题具备时间逻辑。
3. **问题合成**：将筛选出的问题合并为一个新的对比问题，询问事件发生的先后顺序（如：更早/更晚/最早/最晚等），问题应描述清晰且无歧义，保证答案的唯一性。
4. **二次审核**：对生成的问题和答案进行二次审核，确保其符合要求，如符合则can_generate设置为1，否则为0。

### 参考示例
-示例1：
子问题列表：
问题：2023年7月5日温网男子冠军是谁？答案：Jack
问题：2023年7月9日温网女子冠军是谁？答案：Mark
问题：.....
合成问题：
{
    "thoughts": "问题列表中发现有两个子问题均涉及2023年温网赛事，但是信息告知没有明确的时间先后，无法有效生成序数问题",
    "can_generate": 0,
    "question": "",
    "question_date": "",
    "answer": "",
    "temporal_expression_type": "",
    "temporal_scope": "",
    "temporal_granularity": "",
    "temporal_type": "",
    "answer_type": "",
    "reference_document_count": ""
},


-示例2：
问题：2024年6月13日G7峰会在哪里开幕？答案：意大利南部普利亚大区
问题：2023年5月16日G7峰会在哪里举行？答案：日本广岛
合成问题：
{
    "thoughts": "这两个子问题均涉及G7峰会的举办，24年的峰会比23年晚一些。因此可以就此提出问题，询问今年的峰会比去年是不是更晚，这个问题涉及到去年（相对时间），以及对比，所以问题类型设置为composition",
    "can_generate": 1,
    "question": "今年的G7峰会和去年哪个召开时间更晚？",
    "question_date":"2024年6月15日",
    "answer": "今年（2024年）",
    "temporal_expression_type": "implicit",
    "temporal_scope": "long-term",
    "temporal_granularity": "day",
    "temporal_type": "composition",
    "answer_type": "judgement",
    "reference_document_count": "multiple"
}


-示例3
问题：2023年6月15日美国国会通过了什么法案？答案：基础设施法案
问题：2023年9月30日美国国会通过了什么重要预算？答案：专利法案
问题：2024年1月5日美国参议院通过了什么关于税收的法案？答案：2024年减税法案
合成问题：
{
    "thoughts": "问题明确列出了法案，并通过子问题得知哪些法案在2023年通过，哪些是在2024年通过。可以生成一个基于特定年份的对比问题。",
    "can_generate": 1,
    "question": "美国基础设施法案、基础设施法案、减税法案，最早通过的法案是哪个？",
    "question_date":"2024年6月20日",
    "answer": "基础设施法案。",
    "temporal_expression_type": "implicit",
    "temporal_scope": "long-term",
    "temporal_granularity": "day",
    "temporal_type": "ordinal",
    "answer_type": "entity",
    "reference_document_count": "multiple"
}


-示例4
问题：2023年3月10日Meta推出了什么新的社交平台功能？答案：Threads
问题：2024年9月15日Twitter（现X）引入了什么新的订阅功能？答案：付费认证
问题：2024年1月5日TikTok宣布了什么新的商业化工具？答案：广告竞价平台
问题：2023年8月5日TikTok宣布了什么新的大模型？答案：豆包

合成问题
{
    "thoughts": "给定问题阐述了几个产品的发布时间，可以模拟生成一个对比问题，询问哪一个产品的先发布。",
    "can_generate": 1,
    "question": "Twitter的付费认证，Meta的Threads和TikTok的豆包哪一个产品更早推出？",
    "question_date":"2024年9月5日",
    "answer": "Threads",
    "temporal_expression_type": "implicit",
    "temporal_scope": "long-term",
    "temporal_granularity": "day",
    "temporal_type": "ordinal",
    "answer_type": "entity",
    "reference_document_count": "multiple"
}

-示例5：
子问题列表：
问题：2023年1月28日电影《满江红》片方对哪些博主提起了诉讼？答案：沈逸、屠龙的胭脂井、平原公子赵胜和喵斯拉
问题：2023年7月28日有哪些企业入局了流感及带状疱疹疫苗赛道？答案：科兴生物、华兰生物、赛诺菲
问题：.....
合成问题：
{
    "thoughts": "问题列表中的子问题语义相关程度不高，无法有效生成有效合理合情的序数问题，强行生成会导致问题语义不合理，因此设置can_generate为0",
    "can_generate": 0,
    "question": "",
    "question_date": "",
    "answer": "",
    "temporal_expression_type": "",
    "temporal_scope": "",
    "temporal_granularity": "",
    "temporal_type": "",
    "answer_type": "",
    "reference_document_count": ""
},


'''

multi_judgement_expression = '''
### 生成步骤
1. **时间共性筛选**：从子问题列表中筛选出时间相同或者相近的问题（可以是不同粒度上的相同，同一年，同一个月或者同一天）。
2. **提问时间选择**：根据时间相关性合理设置提问时间，确保问题具备时间逻辑。
3. **问题合成**：将筛选出的问题合并为一个新的判断类别的问题，询问多个事件发生是否是在同一时间（年、月、日），问题应描述清晰且无歧义，保证答案的唯一性。
4. **二次审核**：对生成的问题和答案进行二次审核，确保其符合要求，如符合则can_generate设置为1，否则为0。


### 参考示例

-示例1：
子问题列表：
问题：《2022年国民抑郁症蓝皮书》是什么时候发布的？答案：2023年3月21日',
问题：2023年北京市教育委员会发布了什么关于中小学生心理健康的计划？答案：《关于全面加强和改进新时代中小学校学生心理健康工作行动计划（2023—2025年）》',
问题：.....

合成问题：
{
    "can_generate": 1,
    "thoughts": "问题列表中这两个子问题都与心理健康相关文件的发布有关。尽管它们来自不同的组织，但它们的发布时间在同一年（2023年），因此可以合并为一个确定答案的问题，询问两个文件是否在同一年发布。问题的提问时间设定为2023年9月1日，增强了时间敏感性。",
    "question": "北京市教育委员会中小学校学生心理健康工作行动计划和国民抑郁症蓝皮书是同一年发布的吗？",
    "question_date":"2024年9月1日",
    "answer": "是的",
    "temporal_expression_type": "implicit",
    "temporal_scope": "mid-term",
    "temporal_granularity": "year",
    "temporal_type": "direct",
    "answer_type": "judgement",
    "reference_document_count": "multiple"
}

-示例2
问题：2023年7月30日Meta推出了什么新的社交平台功能？答案：Threads
问题：2024年9月15日Twitter（现X）引入了什么新的订阅功能？答案：付费认证
问题：2024年1月5日TikTok宣布了什么新的商业化工具？答案：广告竞价平台
问题：2023年8月5日TikTok宣布了什么新的大模型？答案：豆包

合成问题
{
    "can_generate": 1,
    "thoughts": "给定问题阐述了几个产品的发布时间，可以模拟生成一个判断问题，询问这几个产品是否是同一时间发布，选取时间较为相近的但是不在同一个月的豆包和Threads，增大判断难度。",
    "question": "Meta的Threads和TikTok的豆包，是同一个月发布的吗？",
    "question_date":"2023年12月5日",
    "answer": "不是",
    "temporal_expression_type": "implicit",
    "temporal_scope": "mid-term",
    "temporal_granularity": "month",
    "temporal_type": "direct",
    "answer_type": "judgement",
    "reference_document_count": "multiple"
}

'''

multi_direct_expression = '''
### 生成步骤
1. **筛选共性问题**：从子问题列表中筛选出具有时间相关性或语义共性的问题，例如同一事件的不同时间节点，或不同事件的同一时间特征。
2. **提问时间选择**：合理设定提问时间，确保问题具备时间逻辑，体现时间敏感性。
3. **问题合成**：将筛选出的子问题合并成一个综合性问题，答案包含两个或以上的实体或数字，确保问题的时间因素明确，涉及事件的具体情况、变化或对比，避免模棱两可。
4. **二次审核**：审核生成的问题和答案，确保其准确性、唯一性和完整性。同时，核查问题的语义表达和表述方式是否符合常规用户的提问习惯。若问题符合要求，设置 `can_generate` 为1，否则为0。


### 参考示例

-示例1

 '问题：2023年8月11日哪位球员决定加盟拜仁？答案：凯恩',
 '问题：2024年7月5日拜仁与水晶宫就谁的转会达成协议？答案：奥利斯'
 
合成问题
{
    "can_generate": 1,
    "thoughts": "在提供的子问题中，有两道问题涉及拜仁足球俱乐部在不同时间的球员转会，分别是2023年8月签约凯恩和2024年7月与水晶宫达成奥利斯的转会协议。这两个问题在时间和语义上具有共性，可以合并为一个时间敏感且复杂的问题，询问拜仁在这两次转会中签约的球员。",
    "question": "2023年8月和2024年7月，拜仁分别签约了哪两位球员？",
    "question_date": "2024-09-27",
    "answer": "凯恩和奥利斯",
    "temporal_expression_type": "explicit",
    "temporal_scope": "mid-term",
    "temporal_granularity": "month",
    "temporal_type": "direct",
    "answer_type": "entity",
    "reference_document_count": "multiple"
}


-示例2：
子问题列表：
问题：2023年温网男子冠军是谁？答案：Jack
问题：2023年温网女子冠军是谁？答案：Mark
问题：.....
合成问题：
提问时间：
{
    "can_generate": 1,
    "thoughts": "问题列表中发现有两个子问题均涉及2023年温网赛事，分别询问男子和女子冠军。这两类信息都具有明确答案，时间一致，因此可以合并为一个问题，询问男女冠军的情况。提问时间设置为2024年9月15日，以反映用户对去年赛事的关注。",
    "question": "2023年温网男女子冠军分别是谁？",
    "question_date":"2024年9月15日",
    "answer": "Jack、Mark",
    "temporal_expression_type": "explicit",
    "temporal_scope": "mid-term",
    "temporal_granularity": "year",
    "temporal_type": "direct",
    "answer_type": "entity",
    "reference_document_count": "multiple"
},


-示例3：
问题：2024年6月13日G7峰会在哪里开幕？答案：意大利南部普利亚大区
问题：2023年5月16日G7峰会在哪里举行？答案：日本广岛
合成问题：
{
    "can_generate": 1,
    "thoughts": "这两个子问题均涉及G7峰会的举办地点，分别询问2023年和2024年的会议地点。问题可以合并为一个涵盖两年会议地点的提问，用户能够快速了解连续两年的峰会举办地点。时间范围清晰，问题提问时间为2024年7月15日，反映用户对最近峰会的关注。",
    "question": "近两年的G7峰会在哪里举办？",
    "question_date":"2024年7月15日",
    "answer": "日本广岛和意大利南部普利亚大区",
    "temporal_expression_type": "implicit",
    "temporal_scope": "long-term",
    "temporal_granularity": "year",
    "temporal_type": "relative",
    "answer_type": "entity",
    "reference_document_count": "multiple"
}



-示例4：
问题：2024年6月13日G7峰会在哪里开幕？答案：意大利南部普利亚大区
问题：2023年5月16日G7峰会在哪里举行？答案：日本广岛
合成问题：
{
"can_generate": 1,
"thoughts": "问题列表中多个问题均涉及2024年1月初的股价下跌情况，分别询问Mobileye和苹果的具体跌幅。这两个信息可以合并为一个问题，既能反映时间的敏感性，又能汇总相关数据。问题的提问时间设定为2024年1月6日，以体现用户对这些近期股价表现的关注。",
"question": "2024年1月初苹果和Mobileye的股价分别下跌了多少？",
"question_date":"2024年1月6日",
"answer": "苹果3.6%，Mobileye 29%",
"temporal_expression_type": "explicit",
"temporal_scope": "short-term",
"temporal_granularity": "month",
"temporal_type": "direct",
"answer_type": "entity",
"reference_document_count": "multiple"
}



'''

QA_TYPES = {
    "single_direct_expression": single_direct_expression,
    "single_relative_expression": single_relative_expression,
    "multi_judgement_expression": multi_judgement_expression,
    "multi_direct_expression": multi_direct_expression, 
    "multi_ordinal_expression": multi_ordinal_expression,
}