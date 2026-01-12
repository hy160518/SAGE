# GitHub 提交指南

## 第一步：初始化 Git 仓库

```bash
# 进入项目目录
cd g:\gpt-sage2\0310jianchayuan\sage-paper-reproduction

# 初始化 git 仓库
git init

# 配置用户信息（如果还没配置）
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## 第二步：检查要提交的文件

```bash
# 查看当前文件状态
git status

# 查看 .gitignore 是否正确排除了敏感文件
# 确认以下文件被忽略：
# - configs/api_keys.yaml
# - data/ 目录
# - results/ 目录
# - SAGE_V7/ 论文源文件
```

## 第三步：添加文件到暂存区

```bash
# 添加所有文件
git add .

# 或者选择性添加
git add src/ eval/ configs/ requirements.txt README.md

# 查看暂存区状态
git status
```

## 第四步：提交到本地仓库

```bash
# 创建初始提交
git commit -m "Initial commit: SAGE multimodal forensic data processing framework

- Multi-Agent framework for image/voice/text processing
- UIDN-based entity fusion system
- Comprehensive evaluation modules
- Documentation and configuration templates"
```

## 第五步：在 GitHub 上创建仓库

**选项 A：匿名审稿提交（推荐）**

1. 创建一个**新的 GitHub 账号**（不包含真实姓名）
2. 仓库名称使用通用名称，如：
   - `multimodal-forensic-processing`
   - `sage-framework`
   - `forensic-data-service`

3. 仓库设置：
   - ✅ **Public** （审稿人需要访问）
   - ✅ **不要添加** README/License/gitignore（本地已有）
   - ✅ 仓库描述使用通用说明，不提及论文

**选项 B：正常提交**

1. 登录你的 GitHub 账号
2. 点击 "New repository"
3. 填写仓库信息

## 第六步：关联远程仓库并推送

```bash
# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 或使用 SSH
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# 推送到 GitHub（首次）
git branch -M main
git push -u origin main
```

## 第七步：验证提交

1. 访问你的 GitHub 仓库
2. 检查以下内容：
   - ✅ README.md 显示正确
   - ✅ 代码结构完整
   - ✅ **没有** `configs/api_keys.yaml`（敏感文件）
   - ✅ **没有** `data/` 和 `results/`（大文件/数据）
   - ✅ **没有** `SAGE_V7/`（论文源文件）

## 常见问题

### Q1: 如果不小心提交了敏感文件怎么办？

```bash
# 从 git 历史中完全删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch configs/api_keys.yaml" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push origin --force --all
```

### Q2: 如何创建匿名仓库链接？

**方法 1：使用 Anonymous GitHub**
- 访问 https://anonymous.4open.science/
- 提交你的 GitHub 仓库 URL
- 获得匿名链接用于论文提交

**方法 2：创建私有 Gist**
- 将代码打包成 zip
- 上传到 Dropbox/Google Drive
- 生成匿名分享链接

### Q3: 如何更新已推送的代码？

```bash
# 修改代码后
git add .
git commit -m "Update evaluation metrics system"
git push origin main
```

### Q4: 如何查看提交历史？

```bash
git log --oneline --graph --all
```

## 推荐的提交策略

**匿名审稿期间：**
1. 创建新 GitHub 账号（anonymous-reviewer-xxx）
2. 仓库名称通用化
3. README 不提及作者、机构
4. 使用 Anonymous GitHub 生成匿名链接

**论文接收后：**
1. 迁移到正式账号
2. 更新 README 添加作者信息
3. 添加论文引用
4. 发布 release 版本

## 检查清单

提交前请确认：

- [ ] `.gitignore` 正确配置
- [ ] 敏感文件（API keys）已排除
- [ ] 大文件（数据、模型）已排除
- [ ] README 简洁清晰
- [ ] 代码可运行（至少通过基本测试）
- [ ] 文档完整（配置说明、使用示例）
- [ ] 如果匿名审稿，确保无身份信息

## 后续维护

```bash
# 定期同步
git pull origin main

# 创建功能分支
git checkout -b feature/new-evaluation

# 合并分支
git checkout main
git merge feature/new-evaluation

# 创建标签（版本发布）
git tag -a v1.0.0 -m "Initial release for paper submission"
git push origin v1.0.0
```

---

**重要提醒**：如果是匿名审稿，请使用新账号和 Anonymous GitHub 服务！
