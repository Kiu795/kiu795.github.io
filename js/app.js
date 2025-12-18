const GITHUB_USERNAME = 'Kiu795';
const REPO_NAME = 'kiu795.github.io';

let allPosts = []; // 存储所有文章元数据

// 获取 GitHub 上的文章列表
async function fetchPostsList() {
    try {
        const response = await fetch(`https://api.github.com/repos/${GITHUB_USERNAME}/${REPO_NAME}/contents/posts`);
        if (!response.ok) throw new Error('文章列表加载失败');
        const files = await response.json();
        return files.filter(file => file.name.endsWith('.md'));
    } catch (error) {
        console.error(error);
        return [];
    }
}

// 解析 front matter
function parseFrontMatter(md) {
    const fm = md.match(/^---\s*([\s\S]*?)\s*---/);
    if (!fm) return {};
    const lines = fm[1].split('\n');
    const meta = {};
    lines.forEach(line => {
        const [key, ...rest] = line.split(':');
        if (key && rest) {
            let value = rest.join(':').trim();
            if (value.startsWith('[') && value.endsWith(']')) {
                // 转数组
                value = value.slice(1, -1).split(',').map(v => v.trim());
            }
            meta[key.trim()] = value;
        }
    });
    return meta;
}

// 按日期排序（最新的在前面）
function sortPostsByDate(posts) {
    return [...posts].sort((a, b) => {
        const dateA = new Date(a.date || '1970-01-01');
        const dateB = new Date(b.date || '1970-01-01');
        return dateB - dateA; // 降序排列（最新的在前）
    });
}

// 渲染文章卡片
function renderPosts(posts) {
    const container = document.getElementById('posts-container');
    if (!posts.length) {
        container.innerHTML = `<div class="error-message">❌ 没有匹配的文章</div>`;
        return;
    }

    // 排序后再渲染
    const sortedPosts = sortPostsByDate(posts);

    container.innerHTML = sortedPosts.map(post => `
        <div class="post-card" onclick="location.href='post.html?file=${post.filename}'">
            <div class="post-header">
                <h3 class="post-title">${post.title}</h3>
                <span class="post-date">${post.date || ''}</span>
            </div>
            <div class="post-meta">
                <span class="post-category">${post.category || ''}</span>
                <div class="post-tags">
                    ${(post.tags || []).map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
        </div>
    `).join('');
}

// 初始化文章加载
async function initPosts() {
    const container = document.getElementById('posts-container');
    container.innerHTML = `<div class="loading"><div class="spinner"></div><p>加载文章中...</p></div>`;

    const files = await fetchPostsList();
    const posts = [];

    for (const file of files) {
        const res = await fetch(file.download_url);
        const md = await res.text();
        const meta = parseFrontMatter(md);
        posts.push({ filename: file.name, ...meta });
    }

    allPosts = posts;
    renderPosts(posts);
    
    // 在控制台输出排序结果（方便调试）
    console.log('✅ 文章已加载并按日期排序（最新在前）：');
    console.table(sortPostsByDate(posts).map(p => ({ 
        标题: p.title, 
        日期: p.date 
    })));
}

// 搜索功能
function setupSearch() {
    const input = document.getElementById('searchInput');

    input.addEventListener('input', () => {
        const keyword = input.value.trim().toLowerCase();
        if (!keyword) {
            renderPosts(allPosts);
            return;
        }

        const filtered = allPosts.filter(post => {
            const titleMatch = post.title?.toLowerCase().includes(keyword);
            const categoryMatch = post.category?.toLowerCase().includes(keyword);
            const tagsMatch = (post.tags || []).some(tag => tag.toLowerCase().includes(keyword));
            return titleMatch || categoryMatch || tagsMatch;
        });

        renderPosts(filtered);
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    await initPosts();
    setupSearch();
});