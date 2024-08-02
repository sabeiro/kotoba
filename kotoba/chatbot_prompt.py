import re
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict
from langchain.output_parsers import RegexParser


promptConf = """You are an agent designed to answer questions. You are give a context delimited by triple backticks.
Don't give information not mentioned in the context. If you don't know the answer just say I don't know.
In additon to the answer provide the reason. The reason should be explanation why you think this answer is correct. Use context to generate reason. You may also revise the original input if you think that revising it may ultimately lead to a better response.
It should always be formatted like this:
Answer: string with answer
Confidence: number from 0 to 1
Reason: string with reason
```
{context}
```

Question: {question}
Answer:
Confidence:
Reason:
"""


class AnswerDict(BaseModel):
  answer: str = Field()
  confidence: float = Field()
  reason: str = Field()

class RegexParserConf(RegexParser):
  def parse(self, text:str) -> Dict[str,str]:
    """parse llm output"""
    matchF = re.search(self.regex, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if matchF:
      return {key: matchF.group(i + 1) for i, key in enumerate(self.output_keys)}
    else:
      if self.default_output_key is None:
        raise ValueError(f"could not parse output: {text}")
      else:
        return {
          key: text if key == self.default_output_key else ""
          for key in self.output_keys
          }

parserS = RegexParserConf(regex=r"Answer:\s*(?P<Answer>.*)\s*Confidence:\s*(?P<Confidence>.*)\s*Reason:\s*(?P<Reason>.*)",output_keys=["answer","confidence","reason"])
yesNo = re.compile(r'^\s*(yes|no).*',flags=re.IGNORECASE)
yesRe = re.compile(r'^\s*(yes).*',flags=re.IGNORECASE)

