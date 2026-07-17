module.exports = {
  apps: [
    {
      name: 'pdf-tool',
      script: '/usr/local/bin/gunicorn',
      args: 'app:app --workers 1 --threads 8 --bind 0.0.0.0:3000 --timeout 300 --worker-class gthread',
      cwd: '/home/user/webapp',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1',
        // R2 云端存储配置（通过环境变量传入，留空则使用本地文件模式）
        // R2_BUCKET_NAME: '',
        // R2_ACCOUNT_ID: '',
        // R2_ACCESS_KEY_ID: '',
        // R2_SECRET_ACCESS_KEY: '',
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    }
  ]
}
