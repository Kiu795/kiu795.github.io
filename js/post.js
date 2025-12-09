const GITHUB_USERNAME = 'Kiu795';
const REPO_NAME = `${GITHUB_USERNAME}.github.io`;

// 获取 URL 参数
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

// 渲染文章
async function loadPost() {
    const filename = getQueryParam('file');
    if (!filename) return;

    const container = document.getElementById('post-content');

    try {
        const response = await fetch(`https://raw.githubusercontent.com/${GITHUB_USERNAME}/${REPO_NAME}/main/posts/${filename}`);
        if (!response.ok) throw new Error('文章加载失败');

        const md = await response.text();

        // 移除 front matter
        const content = md.replace(/^---[\s\S]*?---\s*/, '');

        // 渲染 Markdown
        container.innerHTML = marked.parse(content);

        // 高亮代码
        container.querySelectorAll('pre code').forEach(block => {
            hljs.highlightBlock(block);
        });

    } catch (error) {
        container.innerHTML = `<div class="error-message">❌ ${error.message}</div>`;
    }
}

document.addEventListener('DOMContentLoaded', loadPost);
