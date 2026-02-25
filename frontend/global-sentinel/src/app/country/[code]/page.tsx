type Props = { params: Promise<{ code: string }> };

export default async function CountryIntelPage({ params }: Props) {
  const { code } = await params;
  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-52px)] p-8">
      <p className="text-lg text-muted-foreground">
        Country Intel: {code} — PLACEHOLDER
      </p>
    </div>
  );
}
