# Tech Decision Demo

A multi-character technical decision conversation with cross-turn dependencies.
The same team discusses database choice, framework selection, and performance
optimization — with a deliberate pivot when PostgreSQL underperforms mid-project.

## Characters

- **架构师张三** — 系统架构师，技术倾向稳健但有时固执
- **DBA李四** — 数据库管理员，最懂 SQL 性能调优
- **前端王五** — 前端组长，对开发体验非常挑剔

## Conversation Events

1. User: 我们今天立项了，要做一套电商后台系统。架构师张三说数据库先选 PostgreSQL，功能比 MySQL 全。
2. User: DBA李四不太同意，他说 MySQL 8.0 的窗口函数和 CTE 已经很成熟了，运维团队也更熟 MySQL。
3. User: 张三坚持用 PG，说 JSONB 对商品属性灵活建模很重要。李四妥协了，决定用 PostgreSQL 14。
4. User: 前端王五提出框架选型，他倾向 React 18 + TypeScript，说生态更好。
5. User: 张三觉得 Vue 3 学习成本更低，团队里有一半人写过 Vue。争论了一下午。
6. User: 最终前端框架投票选了 Vue 3 + Pinia，王五虽然不太爽但也接受了。
7. User: 张三开始做性能评估，说 PG 的查询计划分析工具很强大，用 EXPLAIN ANALYZE 查出了几个慢查询。
8. User: 李四帮张三加了几个联合索引，QPS 从 200 提到了 800。
9. User: 王五那边也在优化前端，把首页的 bundle 从 2MB 压到了 400KB，首屏加载快了 3 秒。
10. User: 上线第一周，数据库 CPU 持续 90%。李四排查发现 PG 的 autovacuum 在高并发写入场景下频繁触发。
11. User: 张三和李四紧急开了会，李四说 PG 的 MVCC 机制在这种写入密集型场景开销太大。
12. User: 张三终于松口了，同意因为 PG 的 autovacuum 和 MVCC 机制在高并发写入下开销太大，把核心表迁回 MySQL。李四花了两天做了迁移方案。
13. User: 迁移完成后 MySQL 的 CPU 稳定在 40%，写入延迟降了一半。张三对着监控屏沉默了很久。
14. User: 王五吐槽说后端一换数据库，前端十几个 API 的字段类型都要重新对齐，返工了一整周。
15. User: 复盘会议上张三承认当初太固执，以后选型要做 PoC 压测。李四说了一句"早听我的不就好了"。

## Expected EngramRouter Recall

Key cross-turn dependencies:

- Turn 3: 决定用 PostgreSQL（张三力推）
- Turn 12: 因性能问题回退到 MySQL（张三松口）
- Turn 15: 张三承认错误，李四补刀

Expected evidence chains:

- 数据库最终选择: MySQL（Turn 12-15）
- 框架选择: Vue 3（Turn 5-6）
- 性能瓶颈根因: PG 的 autovacuum / MVCC（Turn 10-11）
- 人物立场: 张三→PG, 李四→MySQL, 王五→React（后妥协 Vue）
