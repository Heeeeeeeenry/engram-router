# Bug Investigation Demo

A multi-turn bug investigation scenario where initial hypotheses get disproven
and the true root cause emerges after several dead ends. Designed to test
causal reasoning over scattered timeline evidence.

## Conversation Events

1. User: 今天下午三点，运维群炸了。用户反馈支付页面打不开，报 504 超时。
2. User: 我第一反应是网关挂了。查了 Nginx 日志，发现 upstream 响应时间都在 30 秒以上。
3. User: 顺着 Nginx 查下去，发现是订单服务的 /api/order/create 接口超时。其他接口都正常。
4. User: 我怀疑是数据库慢查询。让 DBA 查了 MySQL 慢日志，结果 /api/order/create 的 SQL 执行只要 50ms。数据库慢查询的假设被排除了。
5. User: 那就不是数据库的问题。我开始怀疑是 Redis 缓存失效导致缓存击穿。
6. User: 查了 Redis 监控，缓存命中率 98%，一切正常。这个假设也被推翻了。
7. User: 这时候已经排查了两个小时，客服那边积压了 200 多个投诉工单。
8. User: 我决定上 Arthas 在线诊断。trace 了一下 /api/order/create 的调用链。
9. User: Arthas 显示 95% 的时间都耗在一个叫 checkInventory 的方法上。这个方法会调第三方仓储系统的 API。
10. User: 真相大白了：第三方仓储系统今天中午 12 点做了升级，API 响应从 100ms 变成了 15 秒。
11. User: 难怪只有 /api/order/create 超时——只有创建订单的时候才需要校验库存。
12. User: 我跟仓储那边的运维通了电话，他们承认改了接口的超时策略但没通知我们。
13. User: 临时修复方案：把 checkInventory 改成异步调用，先创建订单再异步确认库存。半小时上线了热修复。
14. User: 热修复生效后，支付页面恢复正常。客服工单从 200 降到了 30。
15. User: 第二天我写了事故报告。根因是外部依赖变更未通知，我们这边也没加熔断和超时兜底。教训：永远不要信任外部 API 的 SLA。

## Expected EngramRouter Recall

Key cross-turn dependencies:

- Turn 1-2: Bug 发现→初步怀疑网关
- Turn 3-4: 锁定接口→排除数据库
- Turn 5-6: Redis 缓存假设被推翻
- Turn 8-10: Arthas 诊断→根因定位（第三方仓储 API）
- Turn 13-14: 修复→恢复
- Turn 15: 事故复盘

Causal chain: 仓储系统升级(Turn 10) → checkInventory 变慢(Turn 9) → /api/order/create 超时(Turn 3) → 支付页 504(Turn 1)
