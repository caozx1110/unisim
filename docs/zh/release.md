# 发布手册

[English](../en/release.md) | [中文](release.md)

本包以 `unisim-core` 发布并以 `unisim` 导入。生产 PyPI 发布由 `.github/workflows/release.yml` 自动执行，前提是推送了匹配的版本标签，且被标签指向的提交已有成功的 `ci.yml` 运行。工作流使用 GitHub trusted publishing（OIDC）只发布源码发行包；仓库中不保存 PyPI token。它不选择 Python 版本或 OS 矩阵，也绝不发布 wheel。PyPI 发布成功后，工作流的 `github-release` 作业会为该标签创建 GitHub Release，附带 sdist 产物，并以 `CHANGELOG.md` 中对应版本的段落作为发布说明。

仓库管理员必须一次性配置 PyPI trusted publisher：owner 为 `unilabsim`、repository 为 `unisim`、workflow 为 `release.yml`、environment 为 `pypi`。请按期望的审查规则保护该 GitHub environment。

## TestPyPI

1. 更新 `pyproject.toml` 中的 `[project].version`、`CHANGELOG.md`，以及受影响的英文和中文文档。
2. 运行 `make check` 和 `make package`。
3. 运行 `uvx --from twine twine check dist/unisim_core-<version>.tar.gz`。
4. 使用 `~/.pypirc` 中的凭据把 sdist 上传到 TestPyPI；永不打印、复制或提交该文件。
5. 在以 TestPyPI 为包索引的隔离环境中安装精确版本，并验证 `import unisim` 不会加载 `unilab` 或引擎 SDK 模块。
6. 在发布 PR 或发布跟踪 issue 中记录已发布 URL、产物哈希、门禁结果和回滚决定。

在所有命令中使用包自身拥有的发行名和导入名，并永不打印、复制或提交 `~/.pypirc`。

## 生产 PyPI

1. 确认工作树干净、`make check` 通过，并且 changelog 包含发布条目。
2. 等待三个跨平台 `ci.yml` 测试作业、`typecheck` 类型检查作业，以及预发布 sdist 打包作业（现在也等待 `typecheck`）在将要打标签的提交上通过。
3. 创建并推送与包版本精确匹配的 annotated tag，例如 `git tag -a v0.1.13 -m "release: unisim-core 0.1.13"`，随后执行 `git push origin v0.1.13`。发布工作流会校验标签和成功 CI 运行，在 `ubuntu-latest` 上构建并对一个 sdist 做冒烟测试，检查成功后发布该 sdist。Runner 只是执行构建的机器，不约束源码产物。发布时没有 Python 或 OS 矩阵，也不发布 wheel；手动 dispatch 只执行验证，不能发布。
4. 检查工作流、PyPI 产物元数据和已创建的 GitHub Release。失败的运行可以重跑，但已发布版本绝不能覆盖。产物错误时修复源码并发布新的 patch 版本。

## 兼容与移除政策

破坏性移除是发布边界上的硬切：minor 版本可以移除内部接口，而为 UniLab 消费方钉住的稳定公共名称（`unisim.SimBackend`、`unisim.create_backend`、`unisim.ADAPTER_SPECS`、导出的适配器类，以及 `unisim.backend.subprocess_ipc`）只在 major 版本移除。已发布版本绝不回溯修改。允许两类有边界的兼容窗口：

- warn-and-ignore shim（例如仍能解析但会告警的已移除 factory 选项）保留到下一个 major 版本；引入 shim 时在 `CHANGELOG.md` 中写明移除里程碑。
- 持久化数据的 reader 兼容（例如跨越进程边界的历史 pickle state key）在代码中记录产生该格式的版本窗口和具名移除边界——默认为下一个 major 版本。

凡宣布"下一个 major 移除"必须写明版本号，且移除不得早于其具名里程碑发布。
