import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Aziko Notes',

      sidebar: [
        {
          label: 'CUDA',
          items: [
            {
              autogenerate: {
                directory: 'cuda',
              },
            },
          ],
        },
        {
          label: 'AI Infra',
          items: [
            {
              autogenerate: {
                directory: 'ai-infra',
              },
            },
          ],
        },
        {
          label: 'C++',
          items: [
            {
              autogenerate: {
                directory: 'cpp',
              },
            },
          ],
        },
        {
          label: 'Linux',
          items: [
            {
              autogenerate: {
                directory: 'linux',
              },
            },
          ],
        },
      ],
    }),
  ],
});