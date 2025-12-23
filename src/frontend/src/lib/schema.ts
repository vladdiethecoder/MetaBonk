export const HEARTBEAT_SCHEMA_VERSION = Number(import.meta.env.VITE_HEARTBEAT_SCHEMA_VERSION ?? 2);

export const schemaMismatchLabel = (got?: number | null) => {
  const expected = HEARTBEAT_SCHEMA_VERSION;
  const gv = got == null ? "?" : String(got);
  return `expected v${expected}, got v${gv}`;
};
