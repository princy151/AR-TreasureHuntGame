using UnityEngine;

public class TreasureSpawner : MonoBehaviour
{
    public GameObject treasureprefab;      // Your treasure chest prefab
    public int numberOfTreasures = 2;      // How many treasures to spawn
    public float spawnRadius = 3f;         // Radius around player

    void Start()
    {
        SpawnTreasures();
    }

    void SpawnTreasures()
    {
        Vector3 playerPosition = Camera.main.transform.position;

        for (int i = 0; i < numberOfTreasures; i++)
        {
            Vector2 randomPos = Random.insideUnitCircle * spawnRadius;
            Vector3 spawnPos = new Vector3(playerPosition.x + randomPos.x, playerPosition.y - 0.5f, playerPosition.z + randomPos.y);

            Instantiate(treasureprefab, spawnPos, Quaternion.identity);
        }
    }
}
