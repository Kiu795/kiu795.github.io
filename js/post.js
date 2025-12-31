const GITHUB_USERNAME = 'Kiu795';
const REPO_NAME = `kiu795.github.io`;

function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

async function loadPost() {
    const filename = getQueryParam('file');
    const container = document.getElementById('post-content');
    if (!filename) {
        container.innerHTML = `<div class="error-message">❌ 未指定文章文件</div>`;
        return;
    }

    try {
        const response = await fetch(`https://raw.githubusercontent.com/${GITHUB_USERNAME}/${REPO_NAME}/main/posts/${filename}`);
        if (!response.ok) throw new Error('文章加载失败');

        const md = await response.text();
        const content = md.replace(/^---\s*[\s\S]*?\s*---\s*/, ''); // 移除 front matter
        container.innerHTML = marked.parse(content);

        // 高亮代码
        container.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });

        // 渲染 LaTeX 公式
        MathJax.typeset();

    } catch (error) {
        container.innerHTML = `<div class="error-message">❌ ${error.message}</div>`;
    }
}

document.addEventListener('DOMContentLoaded', loadPost);
