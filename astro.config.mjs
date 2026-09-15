import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Aziko Notes',

      sidebar: [
        {
          label: 'CUDA',
          autogenerate: {
            directory: 'cuda',
          },
        },
        {
          label: 'AI Infra',
          autogenerate: {
            directory: 'ai-infra',
          },
        },
        {
          label: 'C++',
          autogenerate: {
            directory: 'cpp',
          },
        },
        {
          label: 'Linux',
          autogenerate: {
            directory: 'linux',
          },
        },
      ],
    }),
  ],
});