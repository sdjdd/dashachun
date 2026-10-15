export const copy = {
  brand: {
    name: "管理控制台",
  },
  auth: {
    login: {
      subtitle: "登录以继续",
      submit: "登录",
      switchPrompt: "还没有账号？",
      switchLink: "注册",
    },
    register: {
      title: "注册账号",
      subtitle: "创建新的账号",
      submit: "注册",
      switchPrompt: "已有账号？",
      switchLink: "登录",
    },
    fields: {
      username: "用户名",
      password: "密码",
    },
    home: {
      greeting: "你好，",
      roles: {
        admin: "管理员",
        user: "用户",
      },
      placeholder: "管理页面开发中",
      logout: "退出登录",
    },
    errors: {
      invalidCredentials: "用户名或密码错误",
      usernameTaken: "用户名已被注册",
      rateLimited: (secs: number) => `尝试过于频繁，请 ${secs} 秒后再试`,
      network: "网络错误，请确认服务已启动后重试",
    },
    validation: {
      usernameRequired: "请输入用户名",
      usernameFormat: "用户名为 3-32 位字母、数字、下划线或连字符",
      passwordRequired: "请输入密码",
      passwordLength: "密码长度为 8-128 位",
    },
  },
  common: {
    requestFailed: (status: number) => `请求失败（HTTP ${status}）`,
  },
};
