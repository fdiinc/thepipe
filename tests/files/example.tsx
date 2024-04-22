// pages/index.tsx
import type {NextPage} from 'next';
import "../app/globals.css";

const Home: NextPage = () => {
  return (
    <>
      <title>The Pipe</title>
      <meta name="description" content="The Pipe - Feeding complex real-world data into large language models." />
      <link rel="icon" href="/favicon.ico" />
    </>
  );
};

export default Home;