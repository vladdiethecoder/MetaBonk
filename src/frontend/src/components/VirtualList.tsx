import type { CSSProperties, ReactElement } from "react";
import { List } from "react-window";

export type ListChildComponentProps = {
  index: number;
  style: CSSProperties;
};

export default function VirtualList({
  height,
  width,
  itemCount,
  itemSize,
  children,
}: {
  height: number;
  width: number;
  itemCount: number;
  itemSize: number;
  children: (props: ListChildComponentProps) => ReactElement;
}) {
  const Row = ({ index, style }: { index: number; style: CSSProperties }) => children({ index, style });

  return <List rowCount={itemCount} rowHeight={itemSize} rowComponent={Row} rowProps={{}} style={{ height, width }} />;
}
