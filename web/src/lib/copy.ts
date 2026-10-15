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
    errors: {
      invalidCredentials: "用户名或密码错误",
      usernameTaken: "用户名已被注册",
      rateLimited: (secs: number) => `尝试过于频繁，请 ${secs} 秒后再试`,
    },
    validation: {
      usernameRequired: "请输入用户名",
      usernameFormat: "用户名为 3-32 位字母、数字、下划线或连字符",
      passwordRequired: "请输入密码",
      passwordLength: "密码长度为 8-128 位",
    },
  },
  shell: {
    logout: "退出登录",
  },
  agents: {
    nav: "智能体",
    title: "智能体",
    description: "每个智能体拥有独立的人设、设备绑定与记忆",
    create: "新建智能体",
    empty: "还没有智能体",
    emptyHint: "点击右上角创建第一个智能体",
    retry: "重试",
    notFound: "未找到该智能体",
    back: "返回列表",
    createdAt: "创建于",
    fields: {
      name: "名称",
      namePlaceholder: "例如：前台小助手",
      persona: "人设提示词",
      personaPlaceholder: "描述这个智能体的角色、语气与行为边界",
      personaOptional: "选填",
    },
    createTitle: "新建智能体",
    createSubmit: "创建",
    editTitle: "基本信息",
    editSubmit: "保存",
    saved: "已保存",
    dangerTitle: "危险区",
    deleteAction: "删除智能体",
    deleteTitle: "删除智能体",
    deleteDescription: (name: string) =>
      `「${name}」的设备绑定将解除，说话人、长期记忆与会话记录将一并删除，且无法恢复。`,
    deleteConfirm: "确认删除",
    cancel: "取消",
    errors: {
      nameTaken: "该名称已被使用",
    },
    validation: {
      nameRequired: "请输入名称",
      nameLength: "名称长度为 1-64 个字符",
      personaLength: "人设提示词最长 4096 个字符",
    },
  },
  common: {
    requestFailed: (status: number) => `请求失败（HTTP ${status}）`,
    network: "网络错误，请确认服务已启动后重试",
  },
};
