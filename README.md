# 项目同步说明

本地已初始化 Git 仓库，并且 `.gitignore` 已过滤数据与日志。按下面步骤推送到 GitHub（以 SSH 和默认分支 main 为例）：

```bash
# 1) 查看当前状态
git status

# 2) 配置提交身份（如尚未设置）
git config user.name "你的名字"
git config user.email "你的邮箱"

# 3) 添加远端（替换为你的仓库地址）
git remote add origin git@github.com:你的用户名/仓库名.git

# 4) 提交
git add .
git commit -m "初始化项目"

# 5) 推送
git branch -M main
git push -u origin main
```

如果使用 HTTPS，将远程地址改为 `https://github.com/你的用户名/仓库名.git`，推送时输入 Token/密码即可。后续更新时，只需重复 `git add`、`git commit`、`git push`。README.md 只含同步说明，如需更详细的项目文档可再补充。***
