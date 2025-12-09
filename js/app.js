// ⚠️ 配置区域 - 你的 GitHub 用户名
const GITHUB_USERNAME = 'Kiu795';
const REPO_NAME = `${GITHUB_USERNAME}.github.io`;

// 首页加载文章列表
async function loadPosts() {
    const container = document.getElementById('posts-container');
    if (!container) return;

    try {
        // 使用 GitHub API 获取 posts 文件夹下的所有文件
        const response = await fetch(
            `https://api.github.com/repos/${GITHUB_USERNAME}/${REPO_NAME}/contents/posts`,
            { headers: { 'Accept': 'application/vnd.github.v3+json' } }
        );

        if (!response.ok) throw new Error('无法加载文章列表');

        const files = await response.json();

        // 过滤出 Markdown 文件并按文件名倒序（最新的在前）
        const mdFiles = files
            .filter(f => f.name.endsWith('.md'))
            .sort((a, b) => b.name.localeCompare(a.name));

        if (mdFiles.length === 0) {
            container.innerHTML = `<div class="error-message">
                <h3>📝 还没有文章</h3>
                <p>在 <code>posts/</code> 目录下添加 Markdown 文件来发布你的第一篇文章吧！</p>
            </div>`;
            return;
        }

        container.innerHTML = '';

        // 遍历所有 Markdown 文件生成文章卡片
        const posts = await Promise.all(mdFiles.map(async file => {
            try {
                const res = await fetch(file.download_url);
                const content = await res.text();
                return parsePost(content, file.name);
            } catch (err) {
                console.error(`加载文章失败: ${file.name}`, err);
                return null;
            }
        }));

        // 渲染文章卡片
        posts.filter(Boolean).forEach(post => {
            const card = createPostCard(post);
            container.appendChild(card);
        });

    } catch (err) {
        container.innerHTML = `<div class="error-message">
            <h3>❌ 加载失败</h3>
            <p>${err.message}</p>
            <p style="font-size:0.9rem;margin-top:1rem;">请检查 js/app.js 中的 GITHUB_USERNAME 是否正确</p>
        </div>`;
    }
}

// 解析 Markdown Front Matter
function parsePost(content, filename) {
    const frontMatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
    const match = content.match(frontMatterRegex);

    let metadata = {
        title: filename.replace('.md', '').replace(/^\d{4}-\d{2}-\d{2}-/, ''),
        date: extractDateFromFilename(filename),
        category: '未分类',
        tags: []
    };
    let body = content;

    if (match) {
        const frontMatter = match[1];
        body = match[2];
        frontMatter.split('\n').forEach(line => {
            const idx = line.indexOf(':');
            if (idx === -1) return;
            const key = line.slice(0, idx).trim();
            const value = line.slice(idx + 1).trim();
            if (key === 'tags') {
                metadata.tags = value.replace(/[\[\]]/g, '').split(',').map(t => t.trim()).filter(Boolean);
            } else metadata[key] = value;
        });
    }

    const excerpt = body.replace(/[#*`\[\]]/g, '').replace(/\n+/g, ' ').slice(0, 200);

    return {
        filename,
        title: metadata.title,
        date: metadata.date,
        category: metadata.category,
        tags: metadata.tags,
        excerpt: excerpt + (excerpt.length >= 200 ? '...' : '')
    };
}

// 从文件名提取日期
function extractDateFromFilename(filename) {
    const match = filename.match(/^(\d{4}-\d{2}-\d{2})/);
    return match ? match[1] : new Date().toISOString().split('T')[0];
}

// 创建文章卡片
function createPostCard(post) {
    const card = document.createElement('div');
    card.className = 'post-card';
    card.onclick = () => viewPost(post.filename);
    card.innerHTML = `
        <div class="post-header">
            <h3 class="post-title">${escapeHtml(post.title)}</h3>
            <span class="post-date">📅 ${post.date}</span>
        </div>
        <p class="post-excerpt">${escapeHtml(post.excerpt)}</p>
        <div class="post-meta">
            <span class="post-category">${escapeHtml(post.category)}</span>
            <div class="post-tags">${post.tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>
        </div>
    `;
    return card;
}

// 跳转文章详情页
function viewPost(filename) {
    window.location.href = `post.html?file=${encodeURIComponent(filename)}`;
}

// HTML 转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 页面加载时执行
document.addEventListener('DOMContentLoaded', loadPosts);
