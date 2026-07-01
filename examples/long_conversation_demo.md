# Long Conversation Demo

This demo is designed to show why lossy summary compression fails and why EngramRouter recalls evidence on demand.

## Conversation Events

1. User: 张三是我前同事，现在在腾讯。
2. User: 我最近一直在看机械键盘。
3. User: 张三前两天送了我一把 HHKB。
4. User: 他说是因为我生日，知道我一直喜欢键盘。
5. User: 今天我们聊别的事情，暂时不提键盘了。
6. User: 很多轮之后，我问：我那个同事送我的键盘是什么牌子？他为什么送？

## Expected EngramRouter Recall

Relevant evidence:

- 张三是我前同事，现在在腾讯。
- 张三前两天送了我一把 HHKB。
- 他说是因为我生日，知道我一直喜欢键盘。

Expected answer support:

- 同事 = 张三
- 公司 = 腾讯
- 礼物品牌 = HHKB
- 原因 = 生日，并且他知道用户喜欢键盘

## Summary Compression Failure

A compressed summary might say:

> 用户和前同事张三聊过礼物和键盘相关内容。

This loses brand, timing and reason.
