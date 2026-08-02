/**
 * PDF Tool Worker — 反向代理入口
 *
 * 将所有请求转发到 BACKEND_URL 环境变量指定的 Flask 后端。
 * 静态资源（/static/*）由 Cloudflare Assets 绑定直接服务。
 */

export interface Env {
  ASSETS: Fetcher;
  BACKEND_URL: string;   // 通过 wrangler secret put BACKEND_URL 设置
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // ── 优先尝试静态资源（/static/*, /favicon.ico 等）──
    // Cloudflare Assets 绑定会自动处理 public/ 目录下的文件
    if (
      url.pathname.startsWith('/static/') ||
      url.pathname === '/favicon.ico' ||
      url.pathname === '/robots.txt'
    ) {
      const assetResponse = await env.ASSETS.fetch(request);
      if (assetResponse.status !== 404) {
        return assetResponse;
      }
    }

    // ── 所有其它请求（包括 /、/api/*）代理到 Flask 后端 ──
    const backendUrl = (env.BACKEND_URL || '').replace(/\/$/, '');

    if (!backendUrl) {
      return new Response(
        JSON.stringify({
          error: 'BACKEND_URL 未配置。请通过 wrangler secret put BACKEND_URL 设置 Flask 后端地址。'
        }),
        {
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }

    // 构造转发 URL
    const targetUrl = backendUrl + url.pathname + url.search;

    // 复制请求（保留 method、headers、body）
    const proxyRequest = new Request(targetUrl, {
      method:  request.method,
      headers: request.headers,
      body:    ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
      // @ts-ignore
      duplex:  'half',
    });

    try {
      const response = await fetch(proxyRequest);

      // 复制响应，加上 CORS 头（允许跨域）
      const headers = new Headers(response.headers);
      headers.set('Access-Control-Allow-Origin', '*');
      headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');

      return new Response(response.body, {
        status:  response.status,
        headers,
      });
    } catch (err: any) {
      return new Response(
        JSON.stringify({ error: `代理请求失败: ${err.message}` }),
        {
          status: 502,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
  },
} satisfies ExportedHandler<Env>;
