const GITHUB_API = 'https://api.github.com/repos/Kiu795/kiu795.github.io/contents/posts';
let allPosts = [];

// 初始化
document.addEventListener("DOMContentLoaded", () => {
    fetchPosts();
});

// 拉取 GitHub 仓库文章
async function fetchPosts() {
    const postsGrid = document.getElementById('postsGrid');
    postsGrid.innerHTML = `<div class="loading">
        <div class="spinner"></div>
        <p>正在加载文章...</p>
    </div>`;
    try {
        const res = await fetch(GITHUB_API);
        const files = await res.json();

        allPosts = await Promise.all(
            files
                .filter(file => file.name.endsWith('.md'))
                .map(async file => {
                    const contentRes = await fetch(file.download_url);
                    const content = await contentRes.text();

                    // 解析头部 metadata
                    const metadata = parseMetadata(content);

                    return {
                        title: metadata.title || file.name.replace('.md', ''),
                        date: metadata.date || '未知',
                        category: metadata.category || '未分类',
                        tags: metadata.tags || [],
                        excerpt: metadata.excerpt || content.slice(0, 120) + '...',
                        content: content
                    };
                })
        );

        renderPosts(allPosts);
    } catch (err) {
        postsGrid.innerHTML = `<div class="error-message">加载文章失败，请稍后再试。</div>`;
        console.error(err);
    }
}

// 解析文章头部 metadata
function parseMetadata(content) {
    const lines = content.split('\n');
    const metadata = {};
    for (let line of lines) {
        if (line.startsWith('title:')) metadata.title = line.replace('title:', '').trim();
        else if (line.startsWith('date:')) metadata.date = line.replace('date:', '').trim();
        else if (line.startsWith('category:')) metadata.category = line.replace('category:', '').trim();
        else if (line.startsWith('tags:')) {
            metadata.tags = line.replace('tags:', '').split(',').map(t => t.trim());
        }
        else if (line.startsWith('excerpt:')) metadata.excerpt = line.replace('excerpt:', '').trim();
        if (line.trim() === '---') break; // 结束 metadata
    }
    return metadata;
}

// 渲染文章卡片
function renderPosts(posts) {
    const postsGrid = document.getElementById('postsGrid');
    if (posts.length === 0) {
        postsGrid.innerHTML = `<p>没有找到相关文章</p>`;
        return;
    }

    postsGrid.innerHTML = '';
    posts.forEach(post => {
        const card = document.createElement('div');
        card.className = 'post-card';
        card.innerHTML = `
            <div class="post-header">
                <h2 class="post-title">${post.title}</h2>
                <span class="post-date">${post.date}</span>
            </div>
            <p class="post-excerpt">${post.excerpt}</p>
            <div class="post-meta">
                <span class="post-category">${post.category}</span>
                <div class="post-tags">
                    ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
        `;
        card.onclick = () => openPostDetail(post);
        postsGrid.appendChild(card);
    });
}

// 分类筛选
function filterByCategory(category) {
    if (category === 'all') renderPosts(allPosts);
    else renderPosts(allPosts.filter(post => post.category === category));
}

// 搜索文章（标题 + 标签）并高亮匹配关键词
function searchPosts() {
    const keyword = document.getElementById('searchInput').value.toLowerCase();
    const filtered = allPosts.filter(post => 
        post.title.toLowerCase().includes(keyword) ||
        post.tags.some(tag => tag.toLowerCase().includes(keyword))
    );

    renderPosts(filtered, keyword);
}

// 点击文章显示详情
function openPostDetail(post) {
    const postsGrid = document.getElementById('postsGrid');
    postsGrid.innerHTML = `
        <div class="post-detail">
            <h1>${post.title}</h1>
            <div class="post-info">
                <span>${post.date}</span>
                <span>${post.category}</span>
                ${post.tags.length ? `<span>标签: ${post.tags.join(', ')}</span>` : ''}
            </div>
            <div class="markdown-content">${marked.parse(post.content)}</div>
            <button onclick="renderPosts(allPosts)" style="margin-top:1rem; padding:0.6rem 1.2rem; border:none; border-radius:6px; background:#333; color:#fff; cursor:pointer;">返回</button>
        </div>
    `;
}

// 渲染文章卡片（支持搜索高亮）
function renderPosts(posts, keyword = '') {
    const postsGrid = document.getElementById('postsGrid');
    if (posts.length === 0) {
        postsGrid.innerHTML = `<p>没有找到相关文章</p>`;
        return;
    }

    postsGrid.innerHTML = '';
    posts.forEach(post => {
        let title = post.title;
        let excerpt = post.excerpt;

        if (keyword) {
            const re = new RegExp(`(${keyword})`, 'gi');
            title = title.replace(re, '<span class="highlight">$1</span>');
            excerpt = excerpt.replace(re, '<span class="highlight">$1</span>');
        }

        const card = document.createElement('div');
        card.className = 'post-card';
        card.innerHTML = `
            <div class="post-header">
                <h2 class="post-title">${title}</h2>
                <span class="post-date">${post.date}</span>
            </div>
            <p class="post-excerpt">${excerpt}</p>
            <div class="post-meta">
                <span class="post-category">${post.category}</span>
                <div class="post-tags">
                    ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
        `;
        card.onclick = () => openPostDetail(post);
        postsGrid.appendChild(card);
    });
}
