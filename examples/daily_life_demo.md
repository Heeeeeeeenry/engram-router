# Daily Life Demo

A 15-turn daily conversation where information about a family member (妈妈) is
scattered across multiple turns. Tests the system's ability to piece together
person attributes, sensory tags, and event memories from fragmented mentions.

## Conversation Events

1. User: 今天下班早，去菜市场买了条鲈鱼。我妈妈说清蒸最好吃。
2. User: 我妈妈蒸鱼有绝活，葱姜丝切得特别细，蒸出来一点腥味都没有。
3. User: 我妈妈做菜的确好吃，我从小到大就没在外面吃过比她做得更好的家常菜。
4. User: 不过我妈妈脾气急，鱼蒸好了必须马上吃，稍微凉一点她就要唠叨。
5. User: 我妈妈年轻时在纺织厂做过十年，后来下岗了就跟着我姥姥学了做菜。
6. User: 我姥姥是四川人，所以我妈妈做菜带点川味，但没那么辣，算改良版。
7. User: 上周末我妈妈做了水煮鱼，放了巨多花椒，我吃到嘴麻。
8. User: 我妈妈的水煮鱼和别人不一样，底下垫的是豆芽和莴笋片，特别爽口。
9. User: 我妈妈今年 62 了，身体还不错，每天早上六点起来去公园打太极。
10. User: 我妈妈退休前在一个中学食堂做主管，管了二十几个师傅，所以脾气急也跟工作有关。
11. User: 我妈妈现在住的房子离我大概三站地铁，周末我常过去蹭饭。
12. User: 昨天她又做了红烧排骨，用冰糖炒的糖色，颜色红亮亮的，我一口气吃了三碗饭。
13. User: 我妈妈的红烧排骨要先焯水去血沫，然后小火焖四十分钟，最后大火收汁。
14. User: 我妈妈不太会玩手机，微信只用来发语音。我教了她好几次视频通话才学会。
15. User: 虽然我妈妈脾气急、爱唠叨，但每次我加班晚了到家，桌上永远扣着一碗热饭。

## Expected EngramRouter Recall

Scattered attributes to piece together:

- 年龄: 62岁（Turn 9）
- 职业: 退休前在中学食堂做主管（Turn 10），之前在纺织厂十年（Turn 5）
- 烹饪风格: 川味改良版（Turn 6），跟姥姥学的（Turn 5）
- 感官标签: 做饭好吃（Turn 3），脾气急（Turn 4, 15），爱唠叨（Turn 4）
- 事件记忆: 清蒸鲈鱼（Turn 1-2），水煮鱼（Turn 7-8），红烧排骨（Turn 12-13）
- 生活习惯: 六点打太极（Turn 9），不会用手机（Turn 14）
- 居住: 离三站地铁（Turn 11）
