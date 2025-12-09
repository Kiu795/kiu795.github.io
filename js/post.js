document.addEventListener("DOMContentLoaded", () => {
    const params = new URLSearchParams(location.search);
    const file = params.get('file');

    if (!file) {
        document.getElementById('post-content').innerHTML = "<p>文章不存在</p>";
        return;
    }

    fetch(`posts/${file}`)
        .then(res => res.text())
        .then(md => {
            document.getElementById('post-content').innerHTML = marked.parse(md);
            document.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
        })
        .catch(err => {
            document.getElementById('post-content').innerHTML = "<p>加载文章失败</p>";
            console.error(err);
        });
});
